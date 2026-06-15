# -*- coding: utf-8 -*-
"""
Round 3: 模型可解释性分析
=========================
加载最佳 DenseNet-121 模型，对验证集样本进行:
  1. Grad-CAM 热力图生成
  2. Grad-CAM++ 精细定位
  3. 按 TP/TN/FP/FN 分类展示
  4. 论文级综合可视化 Figure
  5. SHAP 特征归因分析（可选）

输出: results/interpretability/
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定种子
torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, DataConfig, ModelConfig
from src.dataset import HipProsthesisDataset
from src.models import create_model
from src.interpretability import (
    GradCAM, GradCAMPlusPlus,
    denormalize, find_target_layer,
    _apply_colormap, _overlay_heatmap,
    plot_gradcam_grid, plot_four_panel, plot_summary_figure,
    run_shap_analysis,
)

# ============================================================
# 配置
# ============================================================

DATA_ROOT = "/data/ghaiyan/植入物松动检测/Data"
MODEL_DIR = Path("/data/ghaiyan/植入物松动检测/models")
OUTPUT_DIR = Path("results/interpretability")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

MODEL_TYPE = "densenet121"  # 主要分析最佳模型
MODEL_CKPT_DIR = MODEL_DIR / MODEL_TYPE

# Cost-Sensitive 阈值 (来自 Round 1 最优结果)
COST_SENS_THRESHOLD = 0.35  # 已从 evaluate_results.py 确认

# 每类最多可视化样本数
MAX_SAMPLES_PER_CATEGORY = 5


# ============================================================
# 数据加载
# ============================================================

def build_val_loader(data_root: str, batch_size: int):
    """构建验证集 DataLoader（与 re_evaluate.py 一致）"""
    data_cfg = DataConfig()
    data_cfg.data_root = data_root
    data_cfg.batch_size = batch_size

    val_dataset = HipProsthesisDataset(
        data_dir=str(Path(data_root) / data_cfg.val_dir),
        config=data_cfg,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    # 保留 dataset 引用，用于后续获取原始文件名
    return val_loader, val_dataset


# ============================================================
# 模型加载
# ============================================================

def load_model(model_type: str, checkpoint_dir: Path,
               device: torch.device) -> torch.nn.Module:
    """加载训练好的模型"""
    config = ModelConfig()
    config.model_name = model_type
    config.use_pretrained = False

    model = create_model(config)
    model = model.to(device)

    # 查找最佳 checkpoint
    best_path = checkpoint_dir / "best_model.pth"
    if not best_path.exists():
        candidates = sorted(checkpoint_dir.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
        best_path = candidates[0]

    checkpoint = torch.load(best_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"  Model loaded: {best_path}")
    return model


# ============================================================
# 推理 + 分类
# ============================================================

def run_categorized_inference(
    model, val_loader, gradcam, gradcam_pp,
    device, threshold: float
) -> List[Dict]:
    """
    两阶段推理:
      Phase 1 (no_grad): 快速前向推理，获取预测和分类 (全部样本)
      Phase 2 (with_grad): 只对代表性样本生成 Grad-CAM
    """
    model.eval()

    # ---- Phase 1: 无梯度快速推理 ----
    print("  Phase 1: Fast inference (no_grad)...")
    records = []  # (img_cpu, label, prob_loose, pred, category)
    sample_idx = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1)

            for i in range(images.size(0)):
                label = labels[i].item()
                prob_loose = probs[i, 1].item()
                pred = 1 if prob_loose >= threshold else 0

                if label == 1 and pred == 1:
                    cat = "TP"
                elif label == 0 and pred == 0:
                    cat = "TN"
                elif label == 0 and pred == 1:
                    cat = "FP"
                else:
                    cat = "FN"

                records.append({
                    "img": images[i].cpu(),
                    "label": label,
                    "prob_loose": prob_loose,
                    "pred": pred,
                    "category": cat,
                    "sample_idx": sample_idx,
                })
                sample_idx += 1

    tp = sum(1 for r in records if r['category']=='TP')
    tn = sum(1 for r in records if r['category']=='TN')
    fp = sum(1 for r in records if r['category']=='FP')
    fn = sum(1 for r in records if r['category']=='FN')
    print(f"     TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ---- Phase 2: 仅对代表性样本生成 Grad-CAM ----
    # 选择策略：每类取 top-5 (TP 取最高置信度, FN 取所有, FP 取所有, TN 取随机5个)
    def select_samples(cat_records, n):
        if cat in ("TP", "FN"):
            return sorted(cat_records, key=lambda r: r["prob_loose"], reverse=True)[:n]
        else:
            return sorted(cat_records, key=lambda r: r["prob_loose"])[:n]

    selected = []
    for cat in ["TP", "TN", "FP", "FN"]:
        cat_recs = [r for r in records if r["category"] == cat]
        n = min(MAX_SAMPLES_PER_CATEGORY, len(cat_recs))
        selected.extend(select_samples(cat_recs, n))

    print(f"  Phase 2: Grad-CAM on {len(selected)} representative samples...")

    results = []
    for idx, rec in enumerate(selected):
        print(f"     [{idx+1}/{len(selected)}] {rec['category']} "
              f"(prob={rec['prob_loose']:.3f})")

        img_tensor = rec["img"]
        label = rec["label"]

        gc_overlay = gradcam.generate_overlay(img_tensor, class_idx=label, alpha=0.4)
        gc_pp_overlay = gradcam_pp.generate_overlay(img_tensor, class_idx=label, alpha=0.4)

        # 纯热力图
        x = img_tensor.unsqueeze(0).to(device)
        cam = gradcam(x, class_idx=torch.tensor([label]).to(device))
        cam_np = cam[0].cpu().numpy()
        cam_resized = cv2.resize(cam_np,
                                 (gc_overlay.shape[1], gc_overlay.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        heatmap = _apply_colormap(cam_resized)

        original = denormalize(img_tensor)

        results.append({
            "original": original,
            "gradcam_overlay": gc_overlay,
            "gradcam_pp_overlay": gc_pp_overlay,
            "heatmap": heatmap,
            "label": rec["label"],
            "pred": rec["pred"],
            "prob_loose": rec["prob_loose"],
            "category": rec["category"],
            "sample_idx": rec["sample_idx"],
        })

    print(f"     Done! {len(results)} Grad-CAM visualizations generated.")

    # 返回 Phase 1 的全部记录（用于统计） + Phase 2 的可视化结果
    return results, records


# ============================================================
# 可视化生成
# ============================================================

def categorize_samples(results: List[Dict]) -> Dict[str, List[Dict]]:
    """按 TP/TN/FP/FN 分组"""
    groups = {"TP": [], "TN": [], "FP": [], "FN": []}
    for r in results:
        groups[r["category"]].append(r)
    return groups


def generate_category_visualizations(groups: Dict[str, List[Dict]],
                                      output_dir: str,
                                      max_per_cat: int = 5):
    """为每个类别生成 Grad-CAM 可视化"""

    for cat, samples in groups.items():
        cat_dir = os.path.join(output_dir, "grad_cam", cat.lower())
        os.makedirs(cat_dir, exist_ok=True)

        if not samples:
            print(f"  [{cat}] No samples (0 found)")
            continue

        print(f"  [{cat}] {len(samples)} samples — "
              f"visualizing top {min(max_per_cat, len(samples))}")

        # 按概率排序
        if cat in ("TP", "FN"):
            # Loose 类 — 取概率最高的
            samples_sorted = sorted(samples,
                                    key=lambda x: x["prob_loose"],
                                    reverse=True)
        else:
            # Control 类 — 取概率最低的（即最有信心是 Control）
            samples_sorted = sorted(samples,
                                    key=lambda x: x["prob_loose"])

        selected = samples_sorted[:max_per_cat]

        # 生成每个样本的四面板
        for i, s in enumerate(selected):
            title = (f"{cat} | Prob(Loose)={s['prob_loose']:.3f} | "
                     f"True={'Loose' if s['label']==1 else 'Control'}")
            plot_four_panel(
                s["original"], s["gradcam_overlay"],
                s["gradcam_pp_overlay"], s["heatmap"],
                title,
                save_path=os.path.join(cat_dir, f"{cat}_{i+1}_four_panel.png")
            )

            # 也保存单独的叠加图（用于论文排版）
            plt.imsave(
                os.path.join(cat_dir, f"{cat}_{i+1}_overlay.png"),
                s["gradcam_overlay"]
            )

        # 生成该类的网格图（原图 + Grad-CAM 并排）
        if len(selected) >= 2:
            imgs = [s["original"] for s in selected]
            ovs = [s["gradcam_overlay"] for s in selected]
            titles = [f"{cat}#{j+1}" for j in range(len(selected))]
            plot_gradcam_grid(
                imgs, ovs, titles,
                n_cols=1,
                figsize_per_image=(4, 3),
                suptitle=f"Grad-CAM: {cat} Cases (n={len(samples)})",
                save_path=os.path.join(cat_dir, f"{cat}_grid.png")
            )


def generate_statistics_figure(all_records: List[Dict],
                                cat_counts: Dict[str, int],
                                output_dir: str):
    """生成类别分布统计图"""
    total = sum(cat_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 柱状图
    colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}
    cats = ["TP", "TN", "FP", "FN"]
    bars = axes[0].bar(cats, [cat_counts[c] for c in cats],
                       color=[colors[c] for c in cats], edgecolor="white")
    for bar, count in zip(bars, [cat_counts[c] for c in cats]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{count}\n({count/total*100:.1f}%)",
                     ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title(f"Prediction Categories (Total={total})", fontsize=13)

    # 右: Loose 概率分布 (TP vs FN)
    tp_probs = [r["prob_loose"] for r in all_records if r["category"] == "TP"]
    fn_probs = [r["prob_loose"] for r in all_records if r["category"] == "FN"]

    bins = np.linspace(0, 1, 21)
    if tp_probs:
        axes[1].hist(tp_probs, bins=bins, alpha=0.6, color="#2ecc71",
                     label=f"TP (n={len(tp_probs)})", edgecolor="white")
    if fn_probs:
        axes[1].hist(fn_probs, bins=bins, alpha=0.6, color="#f39c12",
                     label=f"FN (n={len(fn_probs)})", edgecolor="white")

    axes[1].axvline(x=COST_SENS_THRESHOLD, color="red", linestyle="--",
                    linewidth=2, label=f"Threshold={COST_SENS_THRESHOLD}")
    axes[1].set_xlabel("P(Loose)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Loose Class: Probability Distribution", fontsize=13)
    axes[1].legend(fontsize=11)

    plt.suptitle("DenseNet-121: Inference Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "statistics.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  Saved: {save_path}")
    plt.close()


def generate_comparison_gradcam_vs_gradcampp(groups, output_dir):
    """对比 Grad-CAM 和 Grad-CAM++ 在同一 TP 样本上的差异"""
    tp_samples = groups.get("TP", [])
    if not tp_samples:
        return

    # 取前 4 个 TP 样本
    selected = sorted(tp_samples, key=lambda x: x["prob_loose"], reverse=True)[:4]

    fig, axes = plt.subplots(3, len(selected), figsize=(4 * len(selected), 10))

    if len(selected) == 1:
        axes = axes.reshape(3, 1)

    for i, s in enumerate(selected):
        axes[0, i].imshow(s["original"])
        axes[0, i].set_title(f"Original\nP(Loose)={s['prob_loose']:.3f}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(s["gradcam_overlay"])
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(s["gradcam_pp_overlay"])
        axes[2, i].set_title("Grad-CAM++", fontsize=9)
        axes[2, i].axis("off")

    plt.suptitle("Grad-CAM vs Grad-CAM++ Comparison\n"
                 "DenseNet-121 — True Positive Cases",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "gradcam_vs_gradcampp.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  ROUND 3: MODEL INTERPRETABILITY ANALYSIS")
    print(f"  Model: DenseNet-121 (Cost-Sensitive, threshold={COST_SENS_THRESHOLD})")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] Loading validation data...")
    val_loader, val_dataset = build_val_loader(DATA_ROOT, BATCH_SIZE)
    n_loose = sum(1 for _, l, _ in val_dataset if l == 1)
    n_control = sum(1 for _, l, _ in val_dataset if l == 0)
    print(f"  Validation: {len(val_dataset)} samples "
          f"(Control={n_control}, Loose={n_loose})")

    # 2. 加载模型
    print(f"\n[2/5] Loading {MODEL_TYPE} model...")
    model = load_model(MODEL_TYPE, MODEL_CKPT_DIR, DEVICE)
    model.eval()

    # 3. 初始化 Grad-CAM
    print("\n[3/5] Initializing Grad-CAM & Grad-CAM++...")
    target_layer = find_target_layer(model, MODEL_TYPE)
    print(f"  Target layer: {target_layer.__class__.__name__}")

    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMPlusPlus(model, target_layer)

    # 4. 推理 + 生成所有 Grad-CAM
    print(f"\n[4/5] Running inference with Grad-CAM (threshold={COST_SENS_THRESHOLD})...")
    results = run_categorized_inference(
        model, val_loader, gradcam, gradcam_pp,
        DEVICE, COST_SENS_THRESHOLD
    )

    # results = (gradcam_vis_list, all_records_list)
    gradcam_results, all_records = results

    groups = categorize_samples(gradcam_results)

    # 用全部记录做分类统计
    all_cats = Counter(r["category"] for r in all_records)

    # 打印统计
    print("\n  Category Distribution (all samples):")
    for cat in ["TP", "TN", "FP", "FN"]:
        print(f"    {cat}: {all_cats[cat]}")

    # 5. 生成可视化
    print("\n[5/5] Generating visualizations...")

    # 5a. 各类别 Grad-CAM
    print("\n  --- Category Visualizations ---")
    generate_category_visualizations(groups, str(OUTPUT_DIR),
                                      max_per_cat=MAX_SAMPLES_PER_CATEGORY)

    # 5b. 统计图（基于全部样本）
    print("\n  --- Statistics ---")
    generate_statistics_figure(all_records, all_cats, str(OUTPUT_DIR))

    # 5c. Grad-CAM vs Grad-CAM++
    print("\n  --- Grad-CAM vs Grad-CAM++ ---")
    generate_comparison_gradcam_vs_gradcampp(groups, str(OUTPUT_DIR))

    # 5d. 论文级综合 Figure (基于有 Grad-CAM 的样本)
    print("\n  --- Summary Figure ---")
    plot_summary_figure(gradcam_results, os.path.join(OUTPUT_DIR, "summary_figure.png"))

    # 5e. SHAP 分析（可选 — 若 shap 已安装）
    print("\n  --- SHAP Analysis ---")
    # 背景: 从 val_loader 取第一批归一化图像
    bg_batch = next(iter(val_loader))[0][:30].to(DEVICE)
    # TP 测试样本: 用 all_records 中存储的原始归一化 tensor
    tp_tensors = [r["img"] for r in all_records if r["category"] == "TP"][:15]

    if tp_tensors:
        run_shap_analysis(
            model, bg_batch,
            torch.stack(tp_tensors).to(DEVICE),
            DEVICE, str(OUTPUT_DIR),
            n_samples=15
        )

    # 清理 hooks
    gradcam.remove_hooks()
    gradcam_pp.remove_hooks()

    # 保存元数据
    meta = {
        "model": MODEL_TYPE,
        "threshold": COST_SENS_THRESHOLD,
        "n_total": len(all_records),
        "n_control": n_control,
        "n_loose": n_loose,
        "categories": dict(all_cats),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("  ✅ ALL VISUALIZATIONS GENERATED")
    print(f"  📁 Output: {OUTPUT_DIR.absolute()}")
    print("=" * 60)

    # 生成结果总汇
    tp_count = all_cats["TP"]
    fp_count = all_cats["FP"]
    fn_count = all_cats["FN"]
    tn_count = all_cats["TN"]

    print(f"\n  📊 Summary:")
    print(f"     Confusion Matrix:")
    print(f"                 Pred Control  Pred Loose")
    print(f"     True Control      {tn_count:5d}       {fp_count:5d}")
    print(f"     True Loose        {fn_count:5d}       {tp_count:5d}")
    sens = tp_count / max(tp_count + fn_count, 1)
    spec = tn_count / max(tn_count + fp_count, 1)
    print(f"     Sensitivity: {sens:.4f}")
    print(f"     Specificity: {spec:.4f}")


if __name__ == "__main__":
    main()
