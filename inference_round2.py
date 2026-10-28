import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from transformer.config import (
    TrainingConfig,
    DYNASTY_LABELS,
    PLACE_LABELS,
    IMAGERY_LABELS,
    EMOTION_LABELS,
)
from transformer.model import MultiTaskPoetryModel


ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "checkpoints" / "best_model_round2.pt"
DEFAULT_MODEL = "bert-base-chinese"


def _load_label_meta(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data.get("labels", [])}


def _round_float(value: float) -> float:
    return round(float(value), 4)


class Round2PoetryPredictor:
    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_CKPT,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_seq_len: int = 256,
        local_files_only: bool = True,
    ):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_seq_len = max_seq_len
        self.model_name = model_name

        self.dyn_meta = _load_label_meta(ROOT / "labels" / "dynasty_labels.json")
        self.place_meta = _load_label_meta(ROOT / "labels" / "place_labels.json")
        self.imagery_meta = _load_label_meta(ROOT / "labels" / "imagery_labels.json")
        self.emotion_meta = _load_label_meta(ROOT / "labels" / "emotion_labels.json")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        config = TrainingConfig(
            pretrained_model=model_name,
            max_seq_len=max_seq_len,
            device=device,
        )
        self.model = MultiTaskPoetryModel(config).to(device)
        state = torch.load(self.checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text: str, imagery_threshold: float = 0.5) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        dyn_probs = F.softmax(logits["dynasty"], dim=-1)[0].cpu()
        place_probs = F.softmax(logits["place"], dim=-1)[0].cpu()
        imagery_probs = torch.sigmoid(logits["imagery"])[0].cpu()
        emotion_probs = F.softmax(logits["emotion"], dim=-1)[0].cpu()

        dynasty = self._single_label_result(dyn_probs, DYNASTY_LABELS, self.dyn_meta)
        place = self._single_label_result(place_probs, PLACE_LABELS, self.place_meta)
        imagery = self._multi_label_result(
            imagery_probs,
            IMAGERY_LABELS,
            self.imagery_meta,
            imagery_threshold,
        )
        emotion = self._single_label_result(emotion_probs, EMOTION_LABELS, self.emotion_meta)

        return {
            "text": text,
            "dynasty": dynasty,
            "place": place,
            "imagery": imagery,
            "emotion": emotion,
            "confidence": {
                "dynasty": dynasty["confidence"],
                "place": place["confidence"],
                "imagery": imagery["confidence"],
                "emotion": emotion["confidence"],
            },
            "model": {
                "checkpoint": str(self.checkpoint),
                "pretrained_model": self.model_name,
                "device": self.device,
            },
        }

    @staticmethod
    def _single_label_result(
        probs: torch.Tensor,
        label_ids: list[str],
        meta: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        idx = int(torch.argmax(probs).item())
        label_id = label_ids[idx]
        label_meta = meta.get(label_id, {})
        return {
            "id": label_id,
            "label": label_meta.get("zh", label_id),
            "confidence": _round_float(probs[idx].item()),
            "probabilities": {
                label_ids[i]: _round_float(probs[i].item())
                for i in range(len(label_ids))
            },
        }

    @staticmethod
    def _multi_label_result(
        probs: torch.Tensor,
        label_ids: list[str],
        meta: dict[str, dict[str, Any]],
        threshold: float,
    ) -> dict[str, Any]:
        selected = []
        for i, label_id in enumerate(label_ids):
            conf = float(probs[i].item())
            if conf >= threshold:
                label_meta = meta.get(label_id, {})
                selected.append({
                    "id": label_id,
                    "label": label_meta.get("zh", label_id),
                    "confidence": _round_float(conf),
                })
        selected.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "threshold": threshold,
            "labels": selected,
            "confidence": _round_float(max((item["confidence"] for item in selected), default=float(probs.max().item()))),
            "probabilities": {
                label_ids[i]: _round_float(probs[i].item())
                for i in range(len(label_ids))
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Round2 河西走廊诗词单句推理")
    parser.add_argument("text", help="输入一句诗")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--imagery-threshold", type=float, default=0.5)
    parser.add_argument("--allow-download", action="store_true",
                        help="允许 transformers 在本地缓存缺失时访问 HuggingFace")
    args = parser.parse_args()

    predictor = Round2PoetryPredictor(
        checkpoint=args.checkpoint,
        model_name=args.model,
        device=args.device,
        max_seq_len=args.max_len,
        local_files_only=not args.allow_download,
    )
    result = predictor.predict(args.text, imagery_threshold=args.imagery_threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
