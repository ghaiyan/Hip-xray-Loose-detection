# -*- coding: utf-8 -*-
"""
从已完成的训练结果生成所有可视化报告
用户三个模型已训练完毕，此脚本仅需加载checkpoint+history，生成ROC/混淆矩阵/t-SNE等
无需重新训练
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.dataset import create_dataloaders
from src.models import create_model
from src.evaluate import (
    plot_roc_curves, plot_confusion_matrix, plot_training_curves,
    plot_tsne, generate_report, print_report
)


def load_best_model(model_name, config):
    """加载最佳模型权重"""
    checkpoint_dir = Path(config.train.checkpoint_dir)
    best_path = checkpoint_dir / f"baseline_{model_name}_best.pth"
    latest_path = checkpoint_dir / f"baseline_{model_name}_latest.pth"

    ckpt_path = best_path if best_path.exists() else latest_path
    if not ckpt_path.exists():
        print(f"  [WARN] No checkpoint found for {model_name} at {ckpt_path}")
        return None

    print(f"  Loading: {ckpt_path}")

    model = create_model(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_inference(model, dataloader, device):
    """模型推理，收集预测结果"""
    all_preds, all_labels, all_probs = [], [], []
    all_features = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, features = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())

    return {
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "features": np.concatenate(all_features, axis=0),
    }


def read_existing_history(model_name):
    """读取已保存的训练历史"""
    history_path = Path(
        f"models/{model_name}/baseline_{model_name}_history.json"
    )
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/data/ghaiyan/植入物松动检测/Data",
                        help="数据集根目录")
    parser.add_argument("--results_dir", type=str,
                        default="results",
                        help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_inference", action="store_true",
                        help="跳过推理，仅用history中已有的best_val_raw生成图表")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 三个模型名
    model_names = ["resnet50", "densenet121", "efficientnet_b3"]

    # ============================================================
    # 方案A：如果 history 中有 best_val_raw，直接用
    # ============================================================
    all_results_for_roc = {}  # {model_name: {labels, probs, ...}}
    all_summary = {}
    training_curves_data = {}

    use_history_raw = False

    for model_name in model_names:
        hist = read_existing_history(model_name)
        if hist is None:
            print(f"[SKIP] No history file for {model_name}")
            continue

        training_curves_data[model_name] = hist

        raw = hist.get("best_val_raw", {})
        if raw and raw.get("labels") and raw.get("probs"):
            # history 中已有原始预测数据，直接用
            use_history_raw = True
            all_results_for_roc[model_name] = {
                "labels": np.array(raw["labels"]),
                "probs": np.array(raw["probs"]),
                "preds": np.array(raw.get("preds", [])),
            }

        # 提取数值指标
        val_hist = hist["history"]["val"]
        best_idx = np.argmax([h["auc"] for h in val_hist])
        best_metrics = val_hist[best_idx]
        best_metrics["best_epoch"] = hist.get("best_epoch", 0)

        all_summary[model_name] = best_metrics

        # 画训练曲线
        plot_training_curves(hist, results_dir / model_name)
        os.makedirs(results_dir / model_name, exist_ok=True)

    # ============================================================
    # 方案B：如果 history 中没有 raw data，跑一轮推理
    # ============================================================
    if not use_history_raw and not args.skip_inference:
        print("\n[INFO] No raw predictions in history, running inference...")

        config = Config()
        config.data.data_root = args.data_root
        config.data.batch_size = args.batch_size
        config.train.checkpoint_dir = "models"

        dataloaders = create_dataloaders(config.data)
        val_loader = dataloaders["val"]

        for model_name in model_names:
            if model_name not in training_curves_data:
                continue

            config.model.model_name = model_name
            model = load_best_model(model_name, config)
            if model is None:
                continue

            model = model.to(device)
            raw = run_inference(model, val_loader, device)
            all_results_for_roc[model_name] = raw
            all_summary[model_name].update({
                "labels": raw["labels"].tolist(),
                "probs": raw["probs"].tolist(),
                "preds": raw["preds"].tolist(),
            })

    # ============================================================
    # 生成综合可视化
    # ============================================================
    if all_results_for_roc:
        print(f"\n{'='*60}")
        print(f"  生成可视化图表...")
        print(f"{'='*60}")

        # ROC曲线对比
        plot_roc_curves(all_results_for_roc, str(results_dir))
        print("  ✅ ROC curves saved")

        # 各模型混淆矩阵
        for model_name, data in all_results_for_roc.items():
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(data["labels"], data["preds"])
            model_dir = results_dir / model_name
            os.makedirs(model_dir, exist_ok=True)
            plot_confusion_matrix(
                cm, ["Control", "Loose"], str(model_dir),
                title=f"Confusion Matrix - {model_name}"
            )
            print(f"  ✅ Confusion matrix: {model_name}")

        # t-SNE（仅对有features的模型）
        for model_name, data in all_results_for_roc.items():
            if "features" in data and len(data["features"]) > 0:
                plot_tsne(
                    data["features"], data["labels"],
                    ["Control", "Loose"],
                    str(results_dir / model_name),
                    title=f"t-SNE - {model_name}"
                )
                print(f"  ✅ t-SNE: {model_name}")

    # ============================================================
    # 生成报告
    # ============================================================
    if all_summary:
        report = generate_report(all_summary, str(results_dir))
        print_report(report)

        # 保存完整结果
        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summary, f, indent=2, ensure_ascii=False,
                     default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

        print(f"\n✅ All results saved to: {results_dir.resolve()}")

        # 列出生成的文件
        for f in sorted(results_dir.rglob("*")):
            if f.is_file():
                print(f"    {f.relative_to(results_dir)}")
    else:
        print("\n[ERROR] No results found!")


if __name__ == "__main__":
    main()
