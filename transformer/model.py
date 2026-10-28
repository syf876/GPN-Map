"""
transformer/model.py
====================
MultiTaskPoetryModel：BERT 编码器 + 4 个独立分类头

结构：
  bert-base-chinese  (或任意 AutoModel)
  └── [CLS] hidden (768)
      ├── head_dynasty   Linear(768,  3)  → 单标签 CrossEntropy
      ├── head_place     Linear(768, 14)  → 单标签 CrossEntropy
      ├── head_imagery   Linear(768,  6)  → 多标签 BCEWithLogitsLoss
      └── head_emotion   Linear(768,  6)  → 单标签 CrossEntropy

损失加权：
  单标签：L = mean( conf_i * CE(logit_i, label_i) )
  多标签：L = mean( conf_i * mean_c( BCE(logit_i, label_i) ) )
  总损失：Σ task_weight_t * L_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import (
    TrainingConfig,
    DYNASTY_LABELS, PLACE_LABELS, IMAGERY_LABELS, EMOTION_LABELS,
)


class MultiTaskPoetryModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.pretrained_model)
        H = self.encoder.config.hidden_size   # 768 for BERT-base

        self.head_dynasty = nn.Linear(H, len(DYNASTY_LABELS))   # 3
        self.head_place   = nn.Linear(H, len(PLACE_LABELS))     # 14
        self.head_imagery = nn.Linear(H, len(IMAGERY_LABELS))   # 6
        self.head_emotion = nn.Linear(H, len(EMOTION_LABELS))   # 6

        self._task_weights = {
            "dynasty": config.task_weight_dynasty,
            "place":   config.task_weight_place,
            "imagery": config.task_weight_imagery,
            "emotion": config.task_weight_emotion,
        }

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
    ) -> dict[str, torch.Tensor]:
        """返回四任务 logits，不含损失。"""
        cls = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]    # [B, H]

        return {
            "dynasty": self.head_dynasty(cls),   # [B, 3]
            "place":   self.head_place(cls),     # [B, 14]
            "imagery": self.head_imagery(cls),   # [B, 6]
            "emotion": self.head_emotion(cls),   # [B, 6]
        }

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        batch:  dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        计算各任务损失，返回 (total_loss, per_task_loss_dict)。

        dynasty ── 真标签，直接 CrossEntropy，不做置信度加权
        place   ── conf_place 加权 CrossEntropy
        imagery ── conf_imagery 加权 BCEWithLogitsLoss（多标签）
        emotion ── conf_emotion 加权 CrossEntropy（伪标签）
        """
        losses = {}

        # dynasty ── 真标签，无置信度加权，标准 CrossEntropy
        losses["dynasty"] = F.cross_entropy(
            logits["dynasty"], batch["label_dynasty"]
        )

        # place ── conf-weighted CrossEntropy
        losses["place"] = (
            batch["conf_place"] *
            F.cross_entropy(logits["place"], batch["label_place"], reduction="none")
        ).mean()

        # imagery ── conf-weighted BCEWithLogitsLoss（多标签，先按类取均值再按样本加权）
        bce_per_sample = F.binary_cross_entropy_with_logits(
            logits["imagery"], batch["label_imagery"], reduction="none"
        ).mean(dim=1)   # [B]
        losses["imagery"] = (batch["conf_imagery"] * bce_per_sample).mean()

        # emotion ── conf-weighted CrossEntropy
        losses["emotion"] = (
            batch["conf_emotion"] *
            F.cross_entropy(logits["emotion"], batch["label_emotion"], reduction="none")
        ).mean()

        total = sum(self._task_weights[t] * losses[t] for t in losses)
        return total, losses
