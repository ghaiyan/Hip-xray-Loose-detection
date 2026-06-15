# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 - 模型架构
Hip Prosthesis Loosening Detection - Model Architectures

复现文献中的基准模型:
1. ResNet-50 (预训练) - 最常用基线
2. DenseNet-121 - 密集连接，特征复用
3. EfficientNet-B3 - 效率精度平衡
4. Multi-Branch Network - 仿 Guo et al. 2023
5. DL特征提取 + ML分类器 - 仿 Muscato et al. 2023
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import ModelConfig


# ============================================================
# 基础分类器头
# ============================================================

class ClassificationHead(nn.Module):
    """可配置的分类头"""
    def __init__(self, in_features: int, num_classes: int,
                 hidden_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 注意力模块
# ============================================================

class ChannelAttention(nn.Module):
    """SE Channel Attention"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attention)


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================
# 模型1: ResNet-50 (Baseline)
# ============================================================

class HipResNet(nn.Module):
    """基于ResNet的髋关节假体松动检测模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if config.model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.in_features = 2048
        elif config.model_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.in_features = 512
        else:
            raise ValueError(f"Unknown model: {config.model_name}")

        # CBAM注意力
        self.cbam = CBAM(self.in_features)

        # 全局池化和分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassificationHead(
            self.in_features, 2,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate
        )

        if config.freeze_backbone:
            self._freeze_layers()

    def _freeze_layers(self):
        """冻结主干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 2) 分类输出
            features: (B, C) 特征向量（用于t-SNE可视化等）
        """
        features = self.backbone(x)       # (B, C, 7, 7)
        features = self.cbam(features)    # 注意力增强
        pooled = self.avg_pool(features)  # (B, C, 1, 1)
        pooled = torch.flatten(pooled, 1) # (B, C)
        logits = self.classifier(pooled)  # (B, 2)
        return logits, pooled


# ============================================================
# 模型2: DenseNet-121
# ============================================================

class HipDenseNet(nn.Module):
    """基于DenseNet的髋关节假体松动检测模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.in_features = 1024

        # BN + ReLU (DenseNet自带)
        self.norm5 = nn.BatchNorm2d(self.in_features)

        # 注意力
        self.cbam = CBAM(self.in_features)

        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 分类头
        self.classifier = ClassificationHead(
            self.in_features, 2,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate
        )

        if config.freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        features = self.norm5(features)
        features = F.relu(features)
        features = self.cbam(features)
        pooled = self.avg_pool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        return logits, pooled


# ============================================================
# 模型3: EfficientNet-B3
# ============================================================

class HipEfficientNet(nn.Module):
    """基于EfficientNet的髋关节假体松动检测模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.in_features = 1536  # EfficientNet-B3 输出通道

        # 注意力
        self.cbam = CBAM(self.in_features)

        # 池化和分类
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassificationHead(
            self.in_features, 2,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate
        )

        if config.freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        features = self.cbam(features)
        pooled = self.avg_pool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        return logits, pooled


# ============================================================
# 模型4: Multi-Branch Network (仿 Guo et al. 2023)
# ============================================================

class MultiBranchBlock(nn.Module):
    """多分支卷积块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 三个不同尺度的分支
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)
        return self.relu(self.bn(out))


class HipMultiBranch(nn.Module):
    """多分支网络 - 仿Guo et al. 2023:多分支网络检测髋关节术后并发症"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ResNet-50作为特征提取基座
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )

        # 从不同层提取多尺度特征
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # 多分支融合
        self.branch_low = MultiBranchBlock(256, 128)
        self.branch_mid = MultiBranchBlock(512, 128)
        self.branch_high = MultiBranchBlock(1024, 128)

        # 全局信息分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU()
        )

        # 融合层
        total_features = 128 * 3 * 3 + 512  # 多分支+全局
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # 特征提取
        stem_out = self.stem(x)
        f1 = self.layer1(stem_out)   # (B, 256, H/4, W/4)
        f2 = self.layer2(f1)          # (B, 512, H/8, W/8)
        f3 = self.layer3(f2)          # (B, 1024, H/16, W/16)
        f4 = self.layer4(f3)          # (B, 2048, H/32, W/32)

        # 多分支处理
        b_low = self.branch_low(f1)
        b_mid = self.branch_mid(f2)
        b_high = self.branch_high(f3)

        # 全局池化
        b_low_p = F.adaptive_avg_pool2d(b_low, 1).flatten(1)
        b_mid_p = F.adaptive_avg_pool2d(b_mid, 1).flatten(1)
        b_high_p = F.adaptive_avg_pool2d(b_high, 1).flatten(1)

        # 全局特征
        global_feat = self.global_branch(f4)

        # 融合
        combined = torch.cat([b_low_p, b_mid_p, b_high_p, global_feat], dim=1)
        logits = self.fusion(combined)

        return logits, global_feat  # 返回logits和特征向量


# ============================================================
# 模型5: DL特征提取 + ML分类器 (仿 Muscato et al. 2023)
# ============================================================

class DLFeatureExtractor(nn.Module):
    """
    深度学习特征提取器
    提取中间层特征，然后用于传统ML分类器
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 提取到layer3，保留更多空间信息用于特征工程
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # 特征投影
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.projection(x)
        return features


# ============================================================
# 模型工厂
# ============================================================

def create_model(config: ModelConfig) -> nn.Module:
    """模型工厂函数"""
    model_map = {
        "resnet50": HipResNet,
        "resnet34": HipResNet,
        "densenet121": HipDenseNet,
        "efficientnet_b3": HipEfficientNet,
        "multi_branch": HipMultiBranch,
        "dl_feature_extractor": DLFeatureExtractor,
    }

    if config.model_name not in model_map:
        raise ValueError(f"Unknown model: {config.model_name}. "
                         f"Available: {list(model_map.keys())}")

    model_class = model_map[config.model_name]
    return model_class(config)
