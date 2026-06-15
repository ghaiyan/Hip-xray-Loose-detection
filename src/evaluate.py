# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 - 评估与可视化
Hip Prosthesis Loosening Detection - Evaluation & Visualization

包含:
- ROC曲线
- 混淆矩阵
- Grad-CAM热力图
- t-SNE特征可视化
- 训练曲线
- 综合评估报告
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互模式
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (roc_curve, auc, confusion_matrix, accuracy_score,
                              f1_score, recall_score, precision_score,
                              precision_recall_curve, average_precision_score)
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import EvalConfig

# 中文字体配置
rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ============================================================
# 阈值调优 — Youden Index / 代价敏感
# ============================================================

def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray,
                           method: str = "youden",
                           fp_cost: float = 1.0, fn_cost: float = 3.0
                           ) -> tuple:
    """
    在验证集上寻找最优决策阈值

    Args:
        labels: 真实标签 (N,)
        probs: 预测概率 (N, 2), probs[:, 1] 为 Loose 类概率
        method: 调优策略
            - "youden": 最大化 Sensitivity + Specificity - 1
            - "f1": 最大化 F1 分数
            - "cost_sensitive": 最小化代价 = FN * fn_cost + FP * fp_cost
        fp_cost: 误报代价（仅 cost_sensitive）
        fn_cost: 漏诊代价（仅 cost_sensitive，默认 3× 误报）

    Returns:
        best_threshold: 最优阈值
        metrics_at_threshold: 该阈值下的 {sensitivity, specificity, f1, accuracy}
    """
    from sklearn.metrics import (roc_curve, f1_score, accuracy_score,
                                  confusion_matrix)

    loose_probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, loose_probs)

    # 跳过首尾极端阈值 (tpr=0/fpr=0 或 tpr=1/fpr=1)
    valid = (thresholds > 0.001) & (thresholds < 0.999)

    if method == "youden":
        youden = tpr - fpr
        youden = np.where(valid, youden, -np.inf)
        best_idx = np.argmax(youden)
    elif method == "f1":
        f1_scores = np.array([
            f1_score(labels, (loose_probs >= t).astype(int))
            for t in thresholds
        ])
        f1_scores = np.where(valid, f1_scores, -np.inf)
        best_idx = np.argmax(f1_scores)
    elif method == "cost_sensitive":
        costs = []
        for t in thresholds:
            preds = (loose_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            costs.append(fn * fn_cost + fp * fp_cost)
        costs = np.array(costs)
        costs = np.where(valid, costs, np.inf)
        best_idx = np.argmin(costs)
    else:
        raise ValueError(f"Unknown method: {method}")

    best_threshold = float(thresholds[best_idx])
    best_preds = (loose_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, best_preds).ravel()

    metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(labels, best_preds),
        "f1": f1_score(labels, best_preds),
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }

    return best_threshold, metrics


def compute_metrics_with_threshold(labels: np.ndarray, probs: np.ndarray,
                                   threshold: float = 0.5) -> dict:
    """
    使用自定义阈值计算指标（替代 argmax 硬判决）

    Args:
        labels: 真实标签
        probs: 预测概率 (N, 2)
        threshold: Loose 类 (class 1) 的决策阈值
    """
    preds = (probs[:, 1] >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "sensitivity": recall_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "threshold": threshold,
    }

def save_figure(fig: plt.Figure, filename: str, save_dir: str):
    """保存图像到指定目录"""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ============================================================
# ROC曲线
# ============================================================

def plot_roc_curves(results: Dict[str, Dict], save_dir: str):
    """
    绘制多模型ROC曲线对比
    Args:
        results: {model_name: {"labels": [...], "probs": [...]}}
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    for (model_name, data), color in zip(results.items(), colors):
        labels = np.array(data["labels"])
        probs = np.array(data["probs"])

        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model_name} (AUC = {roc_auc:.4f})")

    # 对角线
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC = 0.5)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves - Hip Prosthesis Loosening Detection", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "roc_curves_comparison.png", save_dir)


# ============================================================
# 混淆矩阵
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_dir: str, title: str = "Confusion Matrix",
                          normalize: bool = True):
    """绘制混淆矩阵"""
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_norm = cm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, name in zip(axes,
                               [cm, cm_norm],
                               ["Counts", "Normalized"]):
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name}", fontsize=12)

        # 标注数字
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if name == "Counts":
                    text = f"{int(data[i, j])}"
                else:
                    text = f"{data[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center",
                       fontsize=14, color="white" if data[i, j] > data.max() / 2 else "black")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "confusion_matrix.png", save_dir)


# ============================================================
# 训练曲线
# ============================================================

def plot_training_curves(history: Dict, save_dir: str):
    """绘制训练曲线"""
    train_hist = history["history"]["train"]
    val_hist = history["history"]["val"]
    epochs = range(1, len(train_hist) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h["loss"] for h in train_hist], "b-", label="Train", lw=1.5)
    ax.plot(epochs, [h["loss"] for h in val_hist], "r-", label="Val", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [h["accuracy"] for h in train_hist], "b-", label="Train", lw=1.5)
    ax.plot(epochs, [h["accuracy"] for h in val_hist], "r-", label="Val", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1, 0]
    ax.plot(epochs, [h["auc"] for h in val_hist], "g-", lw=2, label="Val AUC")
    ax.axhline(y=max([h["auc"] for h in val_hist]), color="g", linestyle="--",
               alpha=0.5, label=f"Best: {max([h['auc'] for h in val_hist]):.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sensitivity & Specificity
    ax = axes[1, 1]
    ax.plot(epochs, [h["sensitivity"] for h in val_hist], "orange", label="Sensitivity", lw=1.5)
    ax.plot(epochs, [h["specificity"] for h in val_hist], "purple", label="Specificity", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity & Specificity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "training_curves.png", save_dir)


# ============================================================
# t-SNE可视化
# ============================================================

def plot_tsne(features: np.ndarray, labels: np.ndarray, class_names: List[str],
              save_dir: str, title: str = "t-SNE Feature Visualization"):
    """t-SNE降维可视化"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#2196F3", "#F44336"]
    markers = ["o", "s"]

    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[i], marker=markers[i], label=class_name,
                   alpha=0.6, edgecolors="white", linewidth=0.5, s=50)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)

    save_figure(fig, "tsne_visualization.png", save_dir)


# ============================================================
# 综合评估报告
# ============================================================

def generate_report(results: Dict[str, Dict], save_dir: str) -> Dict:
    """
    生成综合评估报告

    Args:
        results: {model_name: metrics_dict}
        save_dir: 报告保存目录
    """
    report = {
        "title": "Hip Prosthesis Loosening Detection - Model Comparison",
        "models": {}
    }

    for model_name, metrics in results.items():
        model_report = {
            "accuracy": metrics.get("accuracy", 0),
            "auc": metrics.get("auc", 0),
            "f1_score": metrics.get("f1", 0),
            "sensitivity": metrics.get("sensitivity", 0),
            "specificity": metrics.get("specificity", 0),
            "precision": metrics.get("precision", 0),
            "confusion_matrix": {
                "tp": metrics.get("tp", 0),
                "tn": metrics.get("tn", 0),
                "fp": metrics.get("fp", 0),
                "fn": metrics.get("fn", 0),
            }
        }
        report["models"][model_name] = model_report

    # 保存JSON报告
    report_path = Path(save_dir) / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def print_report(report: Dict):
    """打印格式化的评估报告"""
    print("\n" + "=" * 70)
    print(f"  {report['title']}")
    print("=" * 70)

    # 表头
    header = f"{'Model':<25} {'Acc':>7} {'AUC':>7} {'F1':>7} {'Sen':>7} {'Spe':>7}"
    print(header)
    print("-" * 70)

    # 数据行
    for model, metrics in report["models"].items():
        row = (f"{model:<25} "
               f"{metrics['accuracy']:>7.4f} "
               f"{metrics['auc']:>7.4f} "
               f"{metrics['f1_score']:>7.4f} "
               f"{metrics['sensitivity']:>7.4f} "
               f"{metrics['specificity']:>7.4f}")
        print(row)

    print("-" * 70)

    # 最佳模型
    best = max(report["models"].items(), key=lambda x: x[1]["auc"])
    print(f"\n🏆 Best Model: {best[0]} (AUC: {best[1]['auc']:.4f})")
    print("=" * 70)
