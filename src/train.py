# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 - 训练管道
Hip Prosthesis Loosening Detection - Training Pipeline

支持:
- Focal Loss / Cross Entropy Loss
- AdamW / SGD 优化器
- Cosine Annealing / ReduceOnPlateau 调度
- 混合精度训练 (AMP)
- 早停机制
- checkpoint 保存
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              recall_score, confusion_matrix)

from .config import TrainConfig


# ============================================================
# 损失函数
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    FL(pt) = -α_t * (1 - pt)^γ * log(pt)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Focal Loss — 医疗影像专用，对漏诊施加更高惩罚

    核心思想：对不同的类别使用不同的聚焦参数 γ：
    - Control (class 0, 多数类): γ_neg 较大，强力抑制易分样本
    - Loose  (class 1, 少数类): γ_pos = 0，不容忍任何漏诊

    ASL(pt, y) = -α_y * (1 - pt)^(γ_y) * log(pt)

    参考: Ridnik et al., "Asymmetric Loss for Multi-Label Classification", ECCV 2020
    适配为 2-class 单标签版本
    """
    def __init__(self, gamma_neg: float = 3.0, gamma_pos: float = 0.0,
                 alpha_pos: float = 0.75):
        """
        Args:
            gamma_neg: Control 类 (class 0) 的聚焦参数
            gamma_pos: Loose  类 (class 1) 的聚焦参数
            alpha_pos: Loose 类的损失权重 (alpha_neg = 1 - alpha_pos)
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.alpha_pos = alpha_pos   # Loose 权重
        self.alpha_neg = 1.0 - alpha_pos  # Control 权重

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 逐样本 CE loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t ∈ (0, 1]

        # 为每个样本分配对应的 γ 和 α
        gamma_per_sample = torch.where(
            targets == 1,
            torch.tensor(self.gamma_pos, device=inputs.device, dtype=torch.float),
            torch.tensor(self.gamma_neg, device=inputs.device, dtype=torch.float)
        )
        alpha_per_sample = torch.where(
            targets == 1,
            torch.tensor(self.alpha_pos, device=inputs.device, dtype=torch.float),
            torch.tensor(self.alpha_neg, device=inputs.device, dtype=torch.float)
        )

        # ASL = -α_t * (1 - pt)^(γ_t) * log(pt)
        focal_weight = (1.0 - pt) ** gamma_per_sample
        asl_loss = alpha_per_sample * focal_weight * ce_loss
        return asl_loss.mean()


def get_loss_fn(config: TrainConfig) -> nn.Module:
    """获取损失函数"""
    if config.loss == "focal":
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    elif config.loss == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=config.asym_gamma_neg,
            gamma_pos=config.asym_gamma_pos,
            alpha_pos=config.asym_alpha_pos
        )
    else:
        return nn.CrossEntropyLoss()


# ============================================================
# 指标计算
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """
    计算分类指标
    """
    metrics = {}

    # 基础指标
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, average="binary")

    # AUC
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics["auc"] = 0.5

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["sensitivity"] = tp / (tp + fn + 1e-8)  # Recall
    metrics["specificity"] = tn / (tn + fp + 1e-8)
    metrics["precision"] = tp / (tp + fp + 1e-8)
    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)

    return metrics


# ============================================================
# 训练Epoch
# ============================================================

def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                scaler: GradScaler, device: torch.device,
                mixed_precision: bool = True) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if mixed_precision:
            with autocast():
                logits, _ = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    metrics["loss"] = avg_loss
    return metrics


# ============================================================
# 验证Epoch
# ============================================================

@torch.no_grad()
def validate_epoch(model: nn.Module, dataloader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    all_features = []

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits, features = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_features.append(features.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    metrics["loss"] = avg_loss
    metrics["features"] = np.concatenate(all_features, axis=0)
    metrics["probs"] = np.array(all_probs)
    metrics["labels"] = np.array(all_labels)
    metrics["preds"] = np.array(all_preds)

    return metrics


# ============================================================
# 完整训练循环
# ============================================================

class Trainer:
    """训练器"""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 损失函数
        self.criterion = get_loss_fn(config)

        # 优化器
        if config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=config.learning_rate,
                momentum=0.9, weight_decay=config.weight_decay
            )

        # 学习率调度器
        if config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - config.warmup_epochs
            )
        elif config.lr_scheduler == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.lr_step_size,
                gamma=config.lr_gamma
            )

        # 混合精度
        self.scaler = GradScaler() if config.mixed_precision else None

        # 早停
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # 历史记录
        self.history = {"train": [], "val": []}
        self._viz_keys = ["features", "probs", "labels", "preds"]

        # 模型保存路径
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
            "metrics": {k: v for k, v in metrics.items()
                       if k not in ["features", "probs", "labels", "preds"]}
        }

        # 保存最新
        path = self.checkpoint_dir / f"{self.config.__class__.__name__}_latest.pth"
        if hasattr(self, 'experiment_name'):
            path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint, path)

        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)

    def warmup_lr(self, epoch: int):
        """学习率预热"""
        if epoch < self.config.warmup_epochs:
            lr_scale = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale

    def train(self, experiment_name: str = "baseline") -> Dict:
        """完整训练流程"""
        self.experiment_name = experiment_name
        print(f"\n{'='*60}")
        print(f"Starting training: {experiment_name}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # 学习率预热
            self.warmup_lr(epoch)

            # 训练
            train_metrics = train_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.scaler, self.device,
                self.config.mixed_precision
            )

            # 验证
            val_metrics = validate_epoch(
                self.model, self.val_loader, self.criterion, self.device
            )

            # 学习率调度
            if self.config.lr_scheduler == "cosine" and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            elif self.config.lr_scheduler == "reduce_on_plateau":
                self.scheduler.step(val_metrics["auc"])

            # 记录
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            # 打印进度
            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")

        # 提前停止检查 & 保存最佳
            if val_metrics["auc"] > self.best_val_auc + self.config.early_stopping_min_delta:
                self.best_val_auc = val_metrics["auc"]
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
                # 保留最佳epoch的原始预测数据，用于后续ROC/t-SNE等可视化
                self.best_val_raw = {
                    "labels": val_metrics["labels"].tolist(),
                    "probs": val_metrics["probs"].tolist(),
                    "preds": val_metrics["preds"].tolist(),
                }
                print(f"  ✅ New best model! AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n  ⏹ Early stopping at epoch {epoch+1}")
                    break

        # 保存最终模型
        self.save_checkpoint(epoch + 1, val_metrics, is_best=False)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best epoch: {self.best_epoch}, Best Val AUC: {self.best_val_auc:.4f}")
        print(f"{'='*60}")

        # 保存训练历史
        history_path = self.checkpoint_dir / f"{experiment_name}_history.json"
        history_to_save = {
            "experiment_name": experiment_name,
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "best_val_raw": getattr(self, "best_val_raw", {"labels": [], "probs": [], "preds": []}),
            "history": {
                "train": [{k: v for k, v in m.items()
                          if k not in self._viz_keys}
                         for m in self.history["train"]],
                "val": [{k: v for k, v in m.items()
                        if k not in self._viz_keys}
                       for m in self.history["val"]]
            }
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_to_save, f, indent=2, ensure_ascii=False)

        return history_to_save
