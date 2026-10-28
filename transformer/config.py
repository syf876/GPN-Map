"""
transformer/config.py
=====================
标签空间定义 + 训练超参配置
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── 标签空间（与 labels/*.json 保持一致）─────────────────────────────────────
DYNASTY_LABELS = ["DYN_TANG", "DYN_SONG", "DYN_WUDAI"]

PLACE_LABELS = [
    "PLC_YUMENGUAN", "PLC_YANGGUAN",  "PLC_SHAZHOU",   "PLC_DUNHUANG",
    "PLC_GUAZHOU",   "PLC_YUMEN",     "PLC_JIUQUAN",   "PLC_GANZHOU",
    "PLC_WUWEI",     "PLC_LIANGZHOU", "PLC_HEXI",      "PLC_DAMO",
    "PLC_HANHAI",    "PLC_LIUSHA",
]

IMAGERY_LABELS = [
    "IMG_FRONTIER", "IMG_DESERT", "IMG_PASS",
    "IMG_HOMESICK", "IMG_NATURE", "IMG_FEAST",
]

EMOTION_LABELS = [
    "EMO_SORROWFUL", "EMO_BOLD",      "EMO_HOMESICK",
    "EMO_MELANCHOLY","EMO_LEISURELY", "EMO_INDIGNANT",
]

# ── label → int 映射 ─────────────────────────────────────────────────────────
DYN2ID = {l: i for i, l in enumerate(DYNASTY_LABELS)}
PLC2ID = {l: i for i, l in enumerate(PLACE_LABELS)}
IMG2ID = {l: i for i, l in enumerate(IMAGERY_LABELS)}
EMO2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}


@dataclass
class TrainingConfig:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_csv:      Path  = ROOT / "dataset" / "poems_train_round1.csv"

    # ── 模型 ──────────────────────────────────────────────────────────────────
    pretrained_model: str = "bert-base-chinese"
    max_seq_len:    int   = 256

    # ── 训练 ──────────────────────────────────────────────────────────────────
    epochs:         int   = 10
    batch_size:     int   = 32
    lr:             float = 2e-5
    warmup_ratio:   float = 0.1
    weight_decay:   float = 0.01
    grad_clip:      float = 1.0

    # ── 任务损失缩放因子（与 conf 加权独立，控制各任务在总损失中的比重）────────
    task_weight_dynasty:  float = 1.0
    task_weight_place:    float = 1.0
    task_weight_imagery:  float = 1.0
    task_weight_emotion:  float = 0.5   # 伪标签，降权抑制噪声

    # ── 输出 ──────────────────────────────────────────────────────────────────
    output_dir:     Path  = ROOT / "checkpoints"
    log_every:      int   = 20          # steps
    eval_every:     int   = 1           # epochs

    # ── 其他 ──────────────────────────────────────────────────────────────────
    seed:           int   = 42
    device:         str   = "cuda"      # "cuda" | "cpu"
