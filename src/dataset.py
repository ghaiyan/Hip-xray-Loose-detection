# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 - 数据集构建与预处理
Hip Prosthesis Loosening Detection - Dataset & Preprocessing

包含:
1. CLAHE预处理增强X光片对比度
2. 数据增强管道
3. PyTorch Dataset & DataLoader
4. 统计分析工具
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import DataConfig


# ============================================================
# CLAHE 预处理
# ============================================================

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    对X光片应用CLAHE（Contrast Limited Adaptive Histogram Equalization）
    增强局部对比度，突出骨-植入物界面细节
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced = clahe.apply(gray)

    # 转回3通道（为适配预训练模型要求）
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# ============================================================
# 数据集类
# ============================================================

class HipProsthesisDataset(Dataset):
    """
    髋关节假体X光片数据集

    Args:
        data_dir: 数据根目录
        config: 数据配置
        is_train: 是否为训练模式（训练模式启用数据增强）
    """

    def __init__(self, data_dir: str, config: DataConfig, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_train = is_train
        self.class_names = config.class_names
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []
        self._load_samples()

        self.transform = self._build_transform()

    def _load_samples(self):
        """递归加载所有图像样本"""
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"[Warning] Directory not found: {class_dir}")
                continue

            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), class_idx, class_name))

        random.shuffle(self.samples)

    def _build_transform(self):
        """构建图像变换管道"""
        transform_list = []

        if self.is_train:
            transform_list.extend([
                transforms.RandomRotation(degrees=self.config.rotation_degrees),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
            ])

        transform_list.extend([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label, class_name = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)

        # 应用CLAHE增强（可选）
        if self.config.use_clahe and self.is_train:
            if random.random() < 0.8:  # 80%概率应用
                img_array = apply_clahe(
                    img_array,
                    clip_limit=self.config.clip_limit,
                    tile_grid_size=self.config.tile_grid_size
                )
                image = Image.fromarray(img_array)

        # 应用变换
        image = self.transform(image)

        return image, label, class_name


# ============================================================
# 数据加载器工厂
# ============================================================

def create_dataloaders(config: DataConfig) -> Dict[str, DataLoader]:
    """
    创建训练和验证数据加载器

    Returns:
        {"train": DataLoader, "val": DataLoader}
    """
    data_root = Path(config.data_root)

    train_dataset = HipProsthesisDataset(
        data_dir=str(data_root / config.train_dir),
        config=config,
        is_train=True
    )

    val_dataset = HipProsthesisDataset(
        data_dir=str(data_root / config.val_dir),
        config=config,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return {"train": train_loader, "val": val_loader}


def dataset_statistics(config: DataConfig) -> Dict:
    """获取数据集统计信息"""
    data_root = Path(config.data_root)
    stats = {}

    for split in ["train", "val"]:
        split_dir = data_root / getattr(config, f"{split}_dir")
        split_stats = {}
        total = 0
        for class_name in config.class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png")))
                split_stats[class_name] = count
                total += count
            else:
                split_stats[class_name] = 0

        split_stats["total"] = total
        stats[split] = split_stats

    stats["overall_total"] = stats["train"]["total"] + stats["val"]["total"]
    stats["classes"] = config.class_names

    # 计算类别分布比例
    train_control = stats["train"]["Control"]
    train_loose = stats["train"]["Loose"]
    stats["train_ratio"] = {
        "Control": train_control / max(stats["train"]["total"], 1),
        "Loose": train_loose / max(stats["train"]["total"], 1)
    }

    return stats
