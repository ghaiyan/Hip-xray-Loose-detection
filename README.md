# Hip-xray-Loose-detection

> 髋关节植入物松动检测项目，基于 X 光片图像的二分类模型训练、评估与可解释性分析。

## 项目概述

本仓库实现了一个基准模型训练与评估管线，支持以下功能：

- 数据加载与预处理（CLAHE 图像增强、数据增强、归一化）
- 多种深度学习模型架构：ResNet、DenseNet、EfficientNet、多分支网络、DL 特征提取器
- 自定义损失函数：Focal Loss、Asymmetric Loss、CrossEntropy
- 训练：混合精度、学习率调度、早停、checkpoint 保存
- 评估：Accuracy、AUC、F1、Sensitivity、Specificity、ROC 曲线、混淆矩阵、t-SNE
- 可解释性：Grad-CAM / Grad-CAM++ 热力图可视化

## 目录结构

```text
Data/                  # 数据目录
  train/               # 训练集目录
    Control/           # Control 类图像
    Loose/             # Loose 类图像
  val/                 # 验证集目录（同样按照 Control/Loose 分类）

src/                   # 项目源码
  config.py            # 全局配置 dataclass
  dataset.py           # 数据集、CLAHE 与 DataLoader
  models.py            # 模型架构与 create_model 工厂函数
  train.py             # 训练、验证、损失函数、训练循环
  evaluate.py          # 评估指标、绘图与报告生成
  interpretability.py  # Grad-CAM / Grad-CAM++ 与可视化工具
  __init__.py          # 包初始化

train_baseline.py      # 基线训练脚本，支持多模型批量训练与阈值调优
evaluate_results.py    # 基于历史训练结果生成 ROC/混淆矩阵/t-SNE 等可视化
run_interpretability.py# 可解释性分析脚本，生成 Grad-CAM/Grad-CAM++ 结果
requirements.txt      # Python 依赖列表
README.md              # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

主要配置集中在 `src/config.py`：

- `DataConfig`：数据路径、图像尺寸、归一化参数、CLAHE 预处理、数据增强、batch size
- `ModelConfig`：基准模型选择、预训练、分类头、dropout
- `TrainConfig`：训练轮数、学习率、优化器、调度器、损失函数、早停、checkpoint 路径
- `EvalConfig`：评估指标、Grad-CAM 层、可视化结果输出目录

> 默认数据路径为 `/data/ghaiyan/植入物松动检测/Data`，请根据实际环境修改 `data_root` 或脚本参数。

## 使用说明

### 1. 训练基线模型

```bash
python train_baseline.py --loss asymmetric --epochs 50 --batch_size 32 --lr 1e-4 --models resnet50 densenet121 efficientnet_b3
```

脚本说明：
- `--loss`：选择 `focal`、`asymmetric` 或 `cross_entropy`
- `--epochs`：训练轮数
- `--batch_size`：批次大小
- `--lr`：初始学习率
- `--models`：要训练的模型列表

### 2. 生成评估结果

当训练完成后，可通过以下命令生成 ROC 曲线、混淆矩阵和训练曲线：

```bash
python evaluate_results.py --data_root /path/to/Data --results_dir results --batch_size 32 --device cuda
```

如果已有训练历史 `best_val_raw`，脚本会直接加载历史数据生成图表；否则会重新加载 checkpoint 并执行验证推理。

### 3. 可解释性分析

```bash
python run_interpretability.py
```

该脚本默认加载 `densenet121` 模型并生成 Grad-CAM / Grad-CAM++ 热力图，结果输出到 `results/interpretability/`。

## 核心模块说明

### `src/dataset.py`
- `HipProsthesisDataset`：支持训练/验证模式
- 数据增强包含旋转、水平/垂直翻转
- 可选的 CLAHE 亮度对比增强
- `create_dataloaders`：构建训练与验证 `DataLoader`

### `src/models.py`
- `HipResNet`：ResNet 系列基线模型
- `HipDenseNet`：DenseNet-121 模型
- `HipEfficientNet`：EfficientNet-B3 模型
- `HipMultiBranch`：多分支网络架构
- `DLFeatureExtractor`：深度特征提取器
- `create_model(config)`：模型工厂函数，根据 `config.model_name` 创建网络

### `src/train.py`
- `FocalLoss`、`AsymmetricLoss` 与 `CrossEntropyLoss`
- `Trainer`：包含训练循环、验证、学习率调度、混合精度、早停、checkpoint 保存
- `compute_metrics`：精度、AUC、F1、敏感性、特异性等指标

### `src/evaluate.py`
- ROC 曲线、混淆矩阵、训练曲线、t-SNE 可视化
- `generate_report`：生成 JSON 评估报告
- `plot_*` 系列函数用于保存结果图像

### `src/interpretability.py`
- `GradCAM` / `GradCAMPlusPlus`：生成热力图与叠加可视化
- `denormalize`：将归一化图像还原为 RGB
- 其他可视化工具用于结果展示

## 备注

- 该项目针对髋关节植入物松动检测的二分类任务，特别关注少数类 `Loose` 的漏诊风险。
- `AsymmetricLoss` 通过不同类别的聚焦参数，降低 `Loose` 漏诊概率。
- 如果需要在不同机器上运行，请先修改 `src/config.py` 中的 `data_root` 与 `checkpoint_dir`，或通过命令行参数覆盖。
