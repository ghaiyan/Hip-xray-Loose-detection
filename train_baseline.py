# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 - 基准模型批量训练主脚本 (Round 1 优化版)
Hip Prosthesis Loosening Detection - Baseline Training

支持:
- Focal Loss / Asymmetric Loss / Cross Entropy
- 训练后自动阈值调优 (Youden / F1 / 代价敏感)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, DataConfig, ModelConfig, TrainConfig, EvalConfig
from src.dataset import create_dataloaders, dataset_statistics
from src.models import create_model
from src.train import Trainer
from src.evaluate import (plot_roc_curves, plot_confusion_matrix,
                           plot_training_curves, plot_tsne,
                           generate_report, print_report, save_figure,
                           find_optimal_threshold)


def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_system_info(config: Config):
    """打印系统信息"""
    print("\n" + "=" * 70)
    print("  HIP PROSTHESIS LOOSENING DETECTION - BASELINE MODEL TRAINING")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  GPU: None (CPU mode)")

    stats = dataset_statistics(config.data)
    print(f"\n  Dataset Statistics:")
    print(f"    Train - Control: {stats['train']['Control']}, Loose: {stats['train']['Loose']}")
    print(f"    Val   - Control: {stats['val']['Control']},   Loose: {stats['val']['Loose']}")
    print(f"    Total: {stats['overall_total']}")
    print("=" * 70)
    return stats


def run_single_experiment(config: Config) -> dict:
    """运行单个模型实验"""
    print("\nBuilding data loaders...")
    dataloaders = create_dataloaders(config.data)

    print(f"Creating model: {config.model.model_name}...")
    model = create_model(config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config.train
    )

    exp_name = f"{config.train.loss}_{config.model.model_name}"
    history = trainer.train(experiment_name=exp_name)

    return history


def main():
    """主函数：批量训练所有基准模型"""
    parser = argparse.ArgumentParser(description="髋关节植入物松动检测 - 基准模型训练")
    parser.add_argument("--loss", type=str, default="asymmetric",
                        choices=["focal", "asymmetric", "cross_entropy"],
                        help="损失函数类型 (default: asymmetric)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["resnet50", "densenet121", "efficientnet_b3"],
                        help="要训练的模型列表")
    args = parser.parse_args()

    # 基础配置
    base_config = Config()
    base_config.train.epochs = args.epochs
    base_config.train.early_stopping_patience = 10
    base_config.train.loss = args.loss
    base_config.train.learning_rate = args.lr
    base_config.data.batch_size = args.batch_size

    print(f"\n{'#' * 70}")
    print(f"#  ROUND 1 OPTIMIZATION: Loss={args.loss.upper()} + Threshold Tuning")
    print(f"{'#' * 70}")
    if args.loss == "asymmetric":
        print(f"  AsymmetricLoss: gamma_neg={base_config.train.asym_gamma_neg}, "
              f"gamma_pos={base_config.train.asym_gamma_pos}, "
              f"alpha_pos={base_config.train.asym_alpha_pos}")

    # 实验配置列表
    experiments = [
        {"name": name, "lr": args.lr, "dropout": 0.3}
        for name in args.models
    ]

    all_results = {}
    all_tuned_results = {}
    all_histories = {}

    for exp in experiments:
        print(f"\n{'#' * 70}")
        print(f"# EXPERIMENT: {exp['name']}  |  Loss: {args.loss}")
        print(f"{'#' * 70}")

        config = Config()
        config.model.model_name = exp["name"]
        config.model.dropout_rate = exp["dropout"]
        config.model.use_pretrained = True
        config.train.learning_rate = exp["lr"]
        config.train.loss = args.loss
        config.train.checkpoint_dir = f"/data/ghaiyan/植入物松动检测/models/{exp['name']}"
        config.experiment_name = f"{args.loss}_{exp['name']}"

        try:
            history = run_single_experiment(config)

            val_history = history["history"]["val"]
            best_epoch_idx = np.argmax([h["auc"] for h in val_history])
            best_metrics = val_history[best_epoch_idx]

            if "best_val_raw" in history:
                raw = history["best_val_raw"]
                best_metrics["labels"] = raw["labels"]
                best_metrics["probs"] = raw["probs"]
                best_metrics["preds"] = raw["preds"]

            all_results[exp["name"]] = best_metrics
            all_histories[exp["name"]] = history

            print(f"\n✅ {exp['name']} completed. Best AUC: {best_metrics['auc']:.4f}")

        except Exception as e:
            print(f"\n❌ {exp['name']} failed: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # 阈值调优 & 综合报告
    # ============================================================
    if all_results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # ---- 阈值调优 ----
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD TUNING RESULTS")
        print(f"{'=' * 70}")

        tune_methods = ["youden", "f1", "cost_sensitive"]
        tuning_report = {}

        for model_name, metrics in all_results.items():
            if "labels" not in metrics or "probs" not in metrics:
                continue

            labels = np.array(metrics["labels"])
            probs = np.array(metrics["probs"])

            model_tuning = {"default": {
                "threshold": 0.5,
                "sensitivity": metrics.get("sensitivity", 0),
                "specificity": metrics.get("specificity", 0),
                "f1": metrics.get("f1", 0),
                "accuracy": metrics.get("accuracy", 0),
                "fn": metrics.get("fn", 0),
                "fp": metrics.get("fp", 0),
            }}

            for method in tune_methods:
                kwargs = {}
                if method == "cost_sensitive":
                    kwargs = {"fn_cost": 3.0, "fp_cost": 1.0}
                thresh, thresh_metrics = find_optimal_threshold(
                    labels, probs, method=method, **kwargs
                )
                model_tuning[method] = thresh_metrics

            tuning_report[model_name] = model_tuning

        # 打印阈值调优对比表
        header = (f"{'Model':<20} {'Method':<18} {'Thresh':>7} "
                  f"{'Sens':>7} {'Spec':>7} {'F1':>7} {'FN':>5} {'FP':>5}")
        print(header)
        print("-" * 85)

        for exp in experiments:
            name = exp["name"]
            if name not in tuning_report:
                continue
            for method, m in tuning_report[name].items():
                method_label = {
                    "default": "Default (0.5)",
                    "youden": "Youden Index",
                    "f1": "Max F1",
                    "cost_sensitive": "Cost (FNx3)"
                }[method]
                row = (f"{name:<20} {method_label:<18} "
                       f"{m.get('threshold', 0.5):>7.4f} "
                       f"{m.get('sensitivity', 0):>7.4f} "
                       f"{m.get('specificity', 0):>7.4f} "
                       f"{m.get('f1', 0):>7.4f} "
                       f"{m.get('fn', 0):>5d} "
                       f"{m.get('fp', 0):>5d}")
                print(row)
            print("-" * 85)

        # 保存调优报告
        with open(f"{results_dir}/threshold_tuning_report.json", "w", encoding="utf-8") as f:
            json.dump(tuning_report, f, indent=2, ensure_ascii=False)

        # ---- 选择最佳阈值方法（默认用 cost_sensitive） ----
        best_method = "cost_sensitive"
        for exp in experiments:
            name = exp["name"]
            if name not in tuning_report:
                continue
            tm = tuning_report[name][best_method]
            all_tuned_results[name] = {
                **{k: v for k, v in all_results[name].items()
                   if k not in ["features", "probs", "labels", "preds"]},
                "sensitivity": tm["sensitivity"],
                "specificity": tm["specificity"],
                "f1": tm["f1"],
                "accuracy": tm["accuracy"],
                "fn": tm["fn"],
                "fp": tm["fp"],
                "threshold": tm["threshold"],
            }

        # ---- ROC Curve ----
        if len(all_results) > 1:
            plot_roc_curves(all_results, results_dir)

        # ---- Final Report ----
        report = generate_report(all_tuned_results, results_dir)
        report["title"] = f"Hip Prosthesis Loosening Detection - {args.loss.upper()} + {best_method.upper()} Threshold"
        print(f"\n{'=' * 70}")
        print(f"  FINAL REPORT ({args.loss.upper()} + {best_method.upper()} THRESHOLD)")
        print(f"{'=' * 70}")
        print_report(report)

        # 保存所有结果
        with open(f"{results_dir}/all_results.json", "w", encoding="utf-8") as f:
            _viz_keys = ["features", "probs", "labels", "preds"]
            json.dump({
                "loss_type": args.loss,
                "threshold_method": best_method,
                "results_default": {k: {kk: vv for kk, vv in v.items() if kk not in _viz_keys}
                                   for k, v in all_results.items()},
                "results_tuned": all_tuned_results,
                "threshold_tuning": tuning_report,
                "histories": all_histories
            }, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n✅ Results saved to: {results_dir}")
        return all_results, all_tuned_results, all_histories

    return None, None, None


if __name__ == "__main__":
    set_seed(42)
    main()
