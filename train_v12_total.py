import os
import json
import copy
import random
import datetime
import numpy as np

import torch
import torch.nn.functional as F
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Dassl関連
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# ★ ここで "adapters.py" をインポートする (adapters.pyには @TRAINER_REGISTRY.register() された "ADAPTER" がある想定)
from adapters import ADAPTER

# -- シード固定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------------------------------------
# データセットパスの定義
# ----------------------------------------------------------------------------------
IMAGES_DIR = '/mnt/source/datasets/TOTAL_CLIP/images'
LABELS_JSON_PATH = '/mnt/source/datasets/TOTAL_CLIP/labels.json'

# 例として 16クラスを選択 (これを trainer.classnames として使用)
SELECTED_LABELS = [
    "Speed Limit 30 km/h",
    "Main road",
    "Yield",
    "Stop",
    "No vehicles",
    "No entry",
    "Pedestrians",
    "Turn right",
    "Turn left",
    "Only straight",
    "Keep right",
    "Keep left",
    "Road straight arrow marker",
    "Road right arrow marker",
    "Road straight left arrow marker",
    "Road left arrow marker"
]

VAL_RATIO = 0.2
BATCH_SIZE = 32
NUM_WORKERS = 4
BACKBONE = "ViT-B/32"   # CLIPのバックボーン例

# ----------------------------------------------------------------------------------
# Datasetの定義
# ----------------------------------------------------------------------------------
class CustomDataset(Dataset):
    """
    画像とラベルを返す簡易Dataset
    """
    def __init__(self, image_paths, labels, label_to_idx, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.preprocess = preprocess

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)

        numeric_label = self.label_to_idx[self.labels[idx]]
        numeric_label_tensor = torch.tensor(numeric_label, dtype=torch.long)
        return (img_tensor, numeric_label_tensor)

def make_train_val_dataset():
    """
    labels.json を読み込み、
    SELECTED_LABELS に含まれるラベルだけを抽出して
    train/val に分割する
    """
    image_paths, labels = [], []
    with open(LABELS_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ここで SELECTED_LABELS に含まれるラベルだけフィルタ
    for entry in data:
        if entry['label'] in SELECTED_LABELS:
            imgp = os.path.join(IMAGES_DIR, entry['image_name'])
            if os.path.isfile(imgp):
                image_paths.append(imgp)
                labels.append(entry['label'])

    # 万が一、SELECTED_LABELS の画像が0枚ならエラー
    if len(labels) == 0:
        raise ValueError(
            "No images found for the specified SELECTED_LABELS. "
            "Check your dataset or SELECTED_LABELS."
        )

    print(f"Total data for selected labels: {len(labels)}")

    # ラベル→インデックスのマッピング
    label_to_idx = {lbl: i for i, lbl in enumerate(SELECTED_LABELS)}

    # train/val split
    train_idx, val_idx = train_test_split(
        list(range(len(labels))),
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=labels  # ラベル分布を維持
    )

    # CLIPモデルの標準変換を取得
    clip_model, preprocess = clip.load(BACKBONE, device="cpu", jit=False)
    clip_model.eval().float()

    # Dataset作成
    train_dataset = CustomDataset(
        [image_paths[i] for i in train_idx],
        [labels[i]      for i in train_idx],
        label_to_idx,
        preprocess
    )
    val_dataset = CustomDataset(
        [image_paths[i] for i in val_idx],
        [labels[i]      for i in val_idx],
        label_to_idx,
        preprocess
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader

# ----------------------------------------------------------------------------------
# Dassl用のcfgを拡張
# ----------------------------------------------------------------------------------
def extend_cfg_for_adapter(cfg):
    """
    train.py を参考に、TRAINER.ADAPTER や TaskRes ノードを追加
    """
    from yacs.config import CfgNode as CN
    cfg.defrost()
    cfg.CLASSNAMES = SELECTED_LABELS

    if not hasattr(cfg.TRAINER, "ADAPTER"):
        cfg.TRAINER.ADAPTER = CN()
    if not hasattr(cfg.TRAINER, "TaskRes"):
        cfg.TRAINER.TaskRes = CN()
        cfg.TRAINER.TaskRes.ENHANCED_BASE = "none"

    # 初期値
    if not hasattr(cfg.TRAINER.ADAPTER, "INIT"):
        cfg.TRAINER.ADAPTER.INIT = "ZS"
    if not hasattr(cfg.TRAINER.ADAPTER, "CONSTRAINT"):
        cfg.TRAINER.ADAPTER.CONSTRAINT = "l2"
    if not hasattr(cfg.TRAINER.ADAPTER, "ENHANCED_BASE"):
        cfg.TRAINER.ADAPTER.ENHANCED_BASE = "none"
    if not hasattr(cfg.TRAINER.ADAPTER, "PREC"):
        cfg.TRAINER.ADAPTER.PREC = "fp16"

    # 他のデフォルト設定(例: SUBSAMPLE_CLASSES, NUM_SHOTSなど)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.NUM_SHOTS = 1

    cfg.freeze()

def build_cfg_and_trainer(
    adapter_init_mode,
    output_dir="./checkpoints_ClipA",
    max_epoch=10,
    apply_constraint="none",
    enhanced_base="none"
):
    """
    adapters.py 内部で参照する cfg を作成し、返す。
    """
    cfg = get_cfg_default()
    extend_cfg_for_adapter(cfg)
    cfg.defrost()
    cfg.CLASSNAMES = SELECTED_LABELS
    cfg.TRAINER.NAME = "ADAPTER"
    cfg.MODEL.BACKBONE.NAME = BACKBONE
    cfg.TRAINER.ADAPTER.PREC = "fp32"  # "amp" などに変えたい場合はここ
    cfg.TRAINER.ADAPTER.INIT = adapter_init_mode
    cfg.TRAINER.ADAPTER.CONSTRAINT = apply_constraint
    cfg.TRAINER.TaskRes.ENHANCED_BASE = enhanced_base
    cfg.OUTPUT_DIR = output_dir
    cfg.OPTIM.MAX_EPOCH = max_epoch
    cfg.OPTIM.LR = 1e-3
    cfg.OPTIM.OPT = "sgd"

    # adapters.py 側で一応参照するDATASET.NAME
    # Dasslビルトインは使わないが、文字列だけ設定
    cfg.DATASET.NAME = "CIFAR10"

    cfg.freeze()
    return cfg

# ----------------------------------------------------------------------------------
# メイン実行関数
# ----------------------------------------------------------------------------------
def main_run_adapter(
    adapter_init_mode="ClipA",
    output_dir="./checkpoints_ClipA",
    max_epoch=10
):
    """
    指定手法(ClipA, TipA, CrossModalなど)で学習し、model-best.pth.tar を出力。
    """
    print(f"\n==== Start training with adapter_init_mode = {adapter_init_mode} ====")

    # 1) cfg 構築
    cfg = build_cfg_and_trainer(
        adapter_init_mode=adapter_init_mode,
        output_dir=output_dir,
        max_epoch=max_epoch
    )

    # 2) Trainer(ADAPTER) インスタンスを作成
    trainer = ADAPTER(cfg)

    # 3) 学習用/検証用 DataLoader を用意
    train_loader, val_loader = make_train_val_dataset()

    # 4) adapters.py 内の Trainer に DataLoader を渡す
    trainer.train_loader_x = train_loader
    trainer.val_loader = val_loader
    trainer.test_loader = val_loader  # テストデータが無い場合、valを使い回す

    # 5) ★ classnames は SELECTED_LABELS を使用する
    #    これで「self.classnames is empty」のエラーは出なくなる
    trainer.classnames = SELECTED_LABELS

    # 6) 学習実行
    trainer.train()

    # 学習が完了すると、cfg.OUTPUT_DIR に "model-best.pth.tar" が保存される
    print(f"Finished training for {adapter_init_mode}! Weights are in: {cfg.OUTPUT_DIR}\n")


if __name__ == "__main__":
    """
    3つのモードを連続実行するサンプルです。
    実行時間などが気になる場合は、個別に呼び出すと良いでしょう。
    """
    # 1) CLIP-Adapter
    main_run_adapter(
        adapter_init_mode="ClipA",
        output_dir="./checkpoints_ClipA",
        max_epoch=10
    )

    # 2) TIP-Adapter
    main_run_adapter(
        adapter_init_mode="TipA",
        output_dir="./checkpoints_TipA",
        max_epoch=10
    )

    # 3) Cross-Modal Linear Probing
    main_run_adapter(
        adapter_init_mode="CrossModal",
        output_dir="./checkpoints_CrossModal",
        max_epoch=10
    )
