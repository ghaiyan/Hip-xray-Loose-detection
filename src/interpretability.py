# -*- coding: utf-8 -*-
"""
髋关节植入物松动检测 — 模型可解释性分析
Model Interpretability: Grad-CAM, Grad-CAM++, SHAP

用于:
1. 定位模型关注的X光片区域
2. 验证模型是否关注骨-植入物界面等临床相关区域
3. 为论文 Discussion 提供可视化证据
"""

import os
import inspect
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 颜色映射（JET colormap → 出版物友好的 inferno）
# ============================================================

def _apply_colormap(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """将单通道热力图映射为彩色热力图"""
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.4) -> np.ndarray:
    """将热力图叠加到原图上"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay


# ============================================================
# 反向归一化（还原图像用于可视化）
# ============================================================

def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """将 ImageNet 归一化后的 tensor 还原为 RGB 图像 (H, W, 3) uint8"""
    img = tensor.clone().cpu().numpy()
    img = img.transpose(1, 2, 0)  # C,H,W → H,W,C
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    return img


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping

    参考文献:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization." ICCV 2017.

    用法:
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(input_tensor, class_idx=1)  # class_idx=1 → Loose
        heatmap = gradcam.generate_heatmap(cam)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Detect device from model parameters
        self.device = next(model.parameters()).device

        self._forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor,
                 class_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成 Grad-CAM 热力图

        Args:
            x: 输入图像 tensor (B, 3, H, W)，已归一化
            class_idx: 目标类别索引，None 则使用模型预测的类别
                       (0=Control, 1=Loose)

        Returns:
            cam: (B, H', W') 归一化热力图 (0–1)
        """
        self.model.zero_grad()

        # Move input to device
        x = x.to(self.device)

        # Forward
        outputs = self.model(x)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        # Backward — 对目标类别的 logit 求梯度
        one_hot = torch.zeros_like(logits)
        for i, idx in enumerate(class_idx):
            one_hot[i, idx] = 1.0

        logits.backward(gradient=one_hot, retain_graph=True)

        # α_k = 1/Z Σ_i Σ_j ∂y^c / ∂A^k_{ij}
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # L^c = ReLU( Σ_k α_k A^k )
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # 逐样本归一化到 [0, 1]
        b = cam.size(0)
        for i in range(b):
            cam_i = cam[i]
            vmax = cam_i.max()
            if vmax > 0:
                cam[i] = cam_i / vmax

        return cam

    def generate_overlay(self, image_tensor: torch.Tensor,
                         class_idx: int = 1,
                         alpha: float = 0.4,
                         colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        生成 Grad-CAM 叠加图

        Args:
            image_tensor: 单张图像 (3, H, W) 归一化 tensor (CPU or CUDA)
            class_idx: 目标类别
            alpha: 热力图透明度
            colormap: OpenCV colormap

        Returns:
            overlay: (H, W, 3) RGB uint8 叠加图
        """
        x = image_tensor.unsqueeze(0)  # (1, 3, H, W)
        cam = self(x, class_idx=torch.tensor([class_idx]).to(self.device))
        cam_np = cam[0].cpu().numpy()

        # 上采样到原图尺寸
        img_np = denormalize(image_tensor)
        cam_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        heatmap = _apply_colormap(cam_resized, colormap)
        return _overlay_heatmap(img_np, heatmap, alpha)

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


# ============================================================
# Grad-CAM++
# ============================================================

class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: 改进的梯度加权方法

    参考文献:
        Chattopadhyay et al. "Grad-CAM++: Generalized Gradient-based
        Visual Explanations for Deep Convolutional Networks." WACV 2018.

    相比 Grad-CAM 的改进:
    - 对同一物体多个实例提供更好的定位
    - 热力图覆盖更完整的目标区域
    """

    def __call__(self, x: torch.Tensor,
                 class_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.model.zero_grad()

        x = x.to(self.device)
        outputs = self.model(x)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        one_hot = torch.zeros_like(logits)
        for i, idx in enumerate(class_idx):
            one_hot[i, idx] = 1.0

        logits.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ 权重计算
        # α^{kc}_{ij} = (∂²y^c / (∂A^k_{ij})²) /
        #                (2(∂²y^c / (∂A^k_{ij})²) + Σ_a Σ_b A^k_{ab}(∂³y^c / (∂A^k_{ab})³))

        grads = self.gradients  # (B, C, H, W)
        b, c, h, w = grads.size()

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        # Sum over spatial dims
        sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 +
                                sum_activations * grads_power_3 + eps)

        # Positive gradients only
        aij = F.relu(aij)

        # Apply weights
        weights = (aij * grads).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * self.activations).sum(dim=1))

        for i in range(b):
            cam_i = cam[i]
            vmax = cam_i.max()
            if vmax > 0:
                cam[i] = cam_i / vmax

        return cam


# ============================================================
# 目标层发现
# ============================================================

def find_target_layer(model: nn.Module, model_type: str = "densenet121") -> nn.Module:
    """
    自动发现 Grad-CAM 目标层（最后一个有空间输出的卷积层）

    Args:
        model: 模型实例
        model_type: "densenet121" | "resnet50" | "efficientnet_b3"

    Returns:
        target_layer: nn.Module
    """
    if model_type == "densenet121":
        # DenseNet: features → denseblock4 (最后一个 dense block)
        if hasattr(model, "features"):
            # features 是 nn.Sequential, denseblock4 是其中一个子模块
            try:
                return model.features.denseblock4
            except AttributeError:
                # 遍历 features 寻找最后一个 _DenseBlock
                for name, module in reversed(list(model.features.named_children())):
                    if "denseblock" in name.lower():
                        return module
        raise ValueError("Cannot find denseblock4 in DenseNet model")

    elif model_type == "resnet50":
        # ResNet: backbone[-1] 即 layer4 的最后一个 Bottleneck
        if hasattr(model, "backbone"):
            return model.backbone[-1]
        raise ValueError("Cannot find layer4 in ResNet model")

    elif model_type == "efficientnet_b3":
        # EfficientNet: features 的最后一层
        if hasattr(model, "features"):
            # EfficientNet features 最后一个 MBConv
            for name, module in reversed(list(model.features.named_children())):
                if isinstance(module, nn.Conv2d):
                    return module
            # fallback: 最后一个子模块
            children = list(model.features.children())
            if children:
                return children[-1]
        raise ValueError("Cannot find target layer in EfficientNet")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# 可视化工具
# ============================================================

def plot_gradcam_grid(images: List[np.ndarray],
                      overlays: List[np.ndarray],
                      titles: List[str],
                      n_cols: int = 3,
                      figsize_per_image: Tuple[int, int] = (3, 3),
                      suptitle: str = "",
                      save_path: Optional[str] = None):
    """
    绘制 Grad-CAM 结果网格

    每张样本两行: 原图 + Grad-CAM 叠加
    """
    n_samples = len(images)
    n_rows = n_samples

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(figsize_per_image[0] * 2,
                                      figsize_per_image[1] * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_rows):
        # 原图
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"{titles[i]}\n(Original)", fontsize=10)
        axes[i, 0].axis("off")

        # Grad-CAM 叠加
        axes[i, 1].imshow(overlays[i])
        axes[i, 1].set_title(f"{titles[i]}\n(Grad-CAM)", fontsize=10)
        axes[i, 1].axis("off")

    plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")

    plt.close()


def plot_four_panel(original: np.ndarray, grad_cam: np.ndarray,
                    grad_cam_pp: np.ndarray, heatmap_only: np.ndarray,
                    title: str, save_path: Optional[str] = None):
    """
    四面板展示: 原图 | Grad-CAM | Grad-CAM++ | 纯热力图
    适合论文 Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(grad_cam)
    axes[1].set_title("Grad-CAM", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(grad_cam_pp)
    axes[2].set_title("Grad-CAM++", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(heatmap_only)
    axes[3].set_title("Heatmap", fontsize=11)
    axes[3].axis("off")

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")

    plt.close()


def plot_summary_figure(all_results: List[Dict],
                        save_path: str):
    """
    论文级综合可视化 Figure

    布局: 上方 2×2 展示 TP/TN/FP/FN 各一个代表性样本的四面板
          下方: Grad-CAM 区域分布统计条形图
    """
    fig = plt.figure(figsize=(18, 12))

    # --- 上方: 2×2 四面板 ---
    categories = ["TP", "TN", "FP", "FN"]
    titles_cn = {
        "TP": "True Positive\n(Correct: Loose detected)",
        "TN": "True Negative\n(Correct: Control detected)",
        "FP": "False Positive\n(Incorrect: Control→Loose)",
        "FN": "False Negative\n(Missed: Loose→Control)",
    }

    for idx, cat in enumerate(categories):
        samples = [r for r in all_results if r.get("category") == cat]
        if not samples:
            continue

        # 取置信度最高/最低的样本
        sample = samples[0]

        ax_orig = plt.subplot(4, 4, idx + 1)
        ax_orig.imshow(sample["original"])
        ax_orig.set_title(titles_cn[cat], fontsize=9, fontweight="bold")
        ax_orig.axis("off")

        ax_gc = plt.subplot(4, 4, idx + 5)
        ax_gc.imshow(sample["gradcam_overlay"])
        ax_gc.set_title(f"Prob(Loose)={sample['prob_loose']:.3f}", fontsize=8)
        ax_gc.axis("off")

        ax_gcpp = plt.subplot(4, 4, idx + 9)
        ax_gcpp.imshow(sample["gradcam_pp_overlay"])
        ax_gcpp.axis("off")

        ax_hm = plt.subplot(4, 4, idx + 13)
        ax_hm.imshow(sample["heatmap"])
        ax_hm.axis("off")

    plt.suptitle("DenseNet-121: Model Interpretability Analysis\n"
                 "Grad-CAM & Grad-CAM++ Visualization of Hip Prosthesis X-rays",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  Saved summary figure: {save_path}")
    plt.close()


# ============================================================
# SHAP 分析（可选）
# ============================================================

def run_shap_analysis(model: nn.Module,
                      background: torch.Tensor,
                      test_samples: torch.Tensor,
                      device: torch.device,
                      output_dir: str,
                      n_samples: int = 20):
    """
    SHAP GradientExplainer 分析

    Args:
        model: 训练好的模型
        background: 背景样本 (用于 SHAP 期望值估计)，shape (N, 3, H, W)
        test_samples: 待解释样本，shape (M, 3, H, W)
        device: 设备
        output_dir: 输出目录
        n_samples: SHAP 分析的样本数量
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed. Install with: pip install shap")
        print("[SHAP] Skipping SHAP analysis.")
        return

    print("\n" + "=" * 60)
    print("  SHAP Analysis (GradientExplainer)")
    print(f"  SHAP version: {shap.__version__}")
    print("=" * 60)

    model.eval()

    # 限制样本数量
    background = background[:min(50, len(background))]
    test_samples = test_samples[:min(n_samples, len(test_samples))]

    # 创建包装器: 返回完整 2D logits (batch, num_classes)
    # SHAP GradientExplainer 内部用 outputs[:, idx] 做2D索引，
    # 所以模型必须返回 2D tensor，不能只返回单类 logit
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            return logits  # (batch, 2) — 保留完整 2D 输出

    wrapped = ModelWrapper(model).to(device)

    print(f"  Background: {len(background)} samples")
    print(f"  Test: {len(test_samples)} samples")

    # GradientExplainer
    explainer = shap.GradientExplainer(wrapped, background.to(device))

    # 兼容不同 SHAP 版本: 检测 shap_values() 支持的参数
    sig = inspect.signature(explainer.shap_values)
    supported_params = set(sig.parameters.keys())
    print(f"  shap_values supported params: {supported_params}")

    # 只传入支持的参数
    call_kwargs = {}
    if "nsamples" in supported_params:
        call_kwargs["nsamples"] = 100

    test_input = test_samples.to(device)

    try:
        shap_values = explainer.shap_values(test_input, **call_kwargs)
    except TypeError as e:
        # 兜底: 只传位置参数
        print(f"  [WARN] shap_values() call failed with kwargs: {e}")
        print(f"  [WARN] Retrying with positional args only...")
        shap_values = explainer.shap_values(test_input)

    # SHAP GradientExplainer 对多输出模型返回 list:
    #   shap_values[0] → class 0 (Control) 的 SHAP, shape (M, 3, H, W)
    #   shap_values[1] → class 1 (Loose)  的 SHAP, shape (M, 3, H, W)
    # 我们关注 Loose 类，取 index=1
    if isinstance(shap_values, list) and len(shap_values) > 1:
        print(f"  SHAP returned {len(shap_values)} output classes, using class 1 (Loose)")
        shap_loose = np.array(shap_values[1])  # (M, 3, H, W)
    else:
        shap_loose = np.array(shap_values)  # 单输出模型，直接用

    # 对每个通道取平均，生成单通道 summary
    shap_mean = shap_loose.mean(axis=1)  # (M, H, W)

    # 取所有 test sample 的 SHAP 绝对值均值
    shap_abs_mean = np.abs(shap_mean).mean(axis=0)  # (H, W)

    # 保存 SHAP 汇总图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像（第一个 test sample）
    img0 = denormalize(test_samples[0])
    axes[0].imshow(img0)
    axes[0].set_title("Sample Image", fontsize=11)
    axes[0].axis("off")

    # SHAP 单样本（第一个）
    im = axes[1].imshow(shap_mean[0], cmap="RdBu_r", vmin=-np.abs(shap_mean[0]).max(),
                        vmax=np.abs(shap_mean[0]).max())
    axes[1].set_title("SHAP Attribution — Loose class (1st sample)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # SHAP 平均
    im2 = axes[2].imshow(shap_abs_mean, cmap="hot")
    axes[2].set_title("Mean |SHAP| — Loose class (all samples)", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle("SHAP GradientExplainer Analysis\n"
                 "DenseNet-121 — Hip Prosthesis Loosening Detection",
                 fontweight="bold")
    plt.tight_layout()

    shap_path = os.path.join(output_dir, "shap_analysis.png")
    plt.savefig(shap_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  Saved: {shap_path}")
    plt.close()
