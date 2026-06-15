# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 - 全局配置文件
Hip Prosthesis Loosening Detection - Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = r"/data/ghaiyan/植入物松动检测/Data"
    train_dir: str = "train"
    val_dir: str = "val"
    class_names: List[str] = field(default_factory=lambda: ["Control", "Loose"])
    num_classes: int = 2

    # 图像预处理
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # 数据增强（训练集）
    use_random_rotation: bool = True
    rotation_degrees: int = 15
    use_random_flip: bool = True
    use_color_jitter: bool = True
    brightness: float = 0.1
    contrast: float = 0.1

    # CLAHE预处理
    use_clahe: bool = True
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)

    # 数据加载
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    # 基准模型选择
    model_name: str = "resnet50"  # resnet50, densenet121, efficientnet_b3, mobilenet_v3

    # 预训练
    use_pretrained: bool = True
    freeze_backbone: bool = False
    freeze_until_layer: str = ""  # 空则不冻结

    # 分类头
    dropout_rate: float = 0.3
    hidden_dim: int = 512


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw, sgd

    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine, reduce_on_plateau, step
    lr_step_size: int = 15
    lr_gamma: float = 0.1
    warmup_epochs: int = 5

    # 损失函数
    loss: str = "focal"  # focal, cross_entropy, asymmetric
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Asymmetric Loss 参数（仅 loss="asymmetric" 时生效）
    # gamma_neg: Control(class 0) 的聚焦参数，越大越抑制易分负样本
    # gamma_pos: Loose(class 1) 的聚焦参数，设0则不抑制，确保不漏诊
    # alpha_pos: Loose 类的权重（0.5=无偏向，>0.5=偏向Loose）
    asym_gamma_neg: float = 3.0   # 强力抑制易分 Control
    asym_gamma_pos: float = 0.0   # 不抑制 Loose（不容忍漏诊）
    asym_alpha_pos: float = 0.75  # Loose 类损失权重 3倍于 Control

    # 早停
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # 交叉验证
    k_folds: int = 5

    # 保存
    checkpoint_dir: str = r"/data/ghaiyan/植入物松动检测/models"
    save_best_only: bool = True

    # 设备
    device: str = "cuda"  # cuda / cpu
    mixed_precision: bool = True  # 混合精度训练

    # 日志
    log_interval: int = 20
    wandb_enabled: bool = False


@dataclass
class EvalConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "auc", "f1", "sensitivity", "specificity"])
    num_thresholds: int = 200
    grad_cam_layers: List[str] = field(default_factory=lambda: ["layer4", "features.denseblock4"])
    visualization_dir: str = r"/data/ghaiyan/植入物松动检测/results"


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # 实验名称
    experiment_name: str = "baseline_resnet50"

    # 随机种子
    seed: int = 42
