# 隐害先知 - 小样本驱动的农林虫害智检系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

**基于小样本学习的伪装目标检测与识别系统**

[项目简介](#项目简介) • [快速开始](#快速开始) • [模型架构](#模型架构) • [实验结果](#实验结果) • [使用文档](#使用文档)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [实验结果](#实验结果)
- [使用文档](#使用文档)
- [权重文件](#权重文件)
- [常见问题](#常见问题)

---

## 项目简介

本项目实现了一个端到端的农林虫害智能检测系统，针对农林场景中害虫伪装性强、样本稀缺、环境复杂等挑战，提出了一种基于小样本学习的伪装目标检测方法。

### 应用场景

- 🌾 **农作物虫害检测**：水稻、玉米、棉花等作物害虫识别
- 🌲 **林业病虫害监测**：森林病虫害早期预警
- 🚁 **无人机巡检**：航拍图像实时检测
- 📱 **移动端应用**：田间快速诊断

---

## 核心特性

| 特性 | 描述 |
|:----:|:----:|
| 🎯 **小样本学习** | 仅需1-5个样本即可快速适配新虫种，大幅降低标注成本 |
| 🦎 **伪装目标检测** | 针对颜色融合、纹理融合、小目标、遮挡等复杂场景优化 |
| 🔄 **多任务输出** | 同时输出分割掩码、类别标签、边界框，满足多样化需求 |
| ⚡ **实时推理** | 支持35+ FPS的实时检测，适合无人机巡检等场景 |
| 🌍 **跨数据集泛化** | 在不同数据集间迁移能力强，适应新环境 |

---

## 项目结构

```
模型代码/
│
├── 📁 my_models/                    # 模型定义
│   ├── 📁 我的模型/                  # 核心模型
│   │   ├── 📄 model_with_classification.py  # 带分类头的分割模型
│   │   ├── 📄 encoder_decoder.py    # ResNet编码器 + C2F解码器
│   │   ├── 📄 modules.py            # 原型匹配、特征聚合等模块
│   │   ├── 📄 cross_attention.py    # 跨注意力机制
│   │   └── 📄 tta.py                # 测试时增强
│   └── ...
│
├── 📁 checkpoints/                  # 模型权重
│   ├── 📄 demo1_improved_model.pth  # 示例权重
│   └── 📄 README.md                 # 权重说明
│
├── 📁 configs/                      # 配置文件
│   └── 📄 datasets.yaml             # 数据集配置
│
├── 📁 COD10K-v3/                    # COD10K数据集
│
├── 📄 train.py                      # 统一训练脚本
├── 📄 inference.py                  # 推理脚本
├── 📄 generate_weights.py           # 权重生成脚本
│
├── 📄 实验表格数据.md                # 实验结果表格
├── 📄 实验报告.md                    # 实验报告
├── 📄 README.md                     # 项目说明
└── 📄 requirements.txt              # 依赖文件
```

---

## 环境配置

### 系统要求

- 操作系统：Windows 10/11, Linux, macOS
- Python：3.8+
- CUDA：11.0+（GPU加速）
- 内存：8GB+
- GPU显存：4GB+（推荐6GB+）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/pest-detection.git
cd pest-detection

# 2. 创建虚拟环境（推荐）
conda create -n pest python=3.8
conda activate pest

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(torch.__version__)"
```

### 主要依赖

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.21.0
tqdm>=4.60.0
```

---

## 快速开始

### 1️⃣ 准备数据集

将数据集放置在相应目录：

```
COD10K-v3/
├── Train/
│   ├── Image/       # 训练图像
│   └── GT_Object/   # 分割标注
└── Test/
    ├── Image/       # 测试图像
    └── GT_Object/   # 测试标注
```

### 2️⃣ 训练模型

```bash
# 小样本训练 (5-shot)
python train.py --mode fewshot --shot 5 --dataset COD10K --epochs 50

# 全监督训练
python train.py --mode full --dataset IP102 --epochs 100

# 消融实验
python train.py --mode ablation --ablation_id 1 --epochs 50
```

### 3️⃣ 模型推理

```bash
# 单张图像推理
python inference.py --image test.jpg --weights checkpoints/model_final.pth

# 批量推理
python inference.py --folder ./images --weights checkpoints/model_final.pth

# 视频推理
python inference.py --video test.mp4 --weights checkpoints/model_final.pth
```

### 4️⃣ 查看结果

推理结果保存在 `output/` 目录：
- `*_result.jpg` - 可视化结果
- `results.json` - 检测结果详情

---

## 模型架构

### 整体架构

```
输入图像
    ↓
┌─────────────────────────────────────┐
│  ResNet50 编码器                      │
│  - 多尺度特征提取                      │
│  - 预训练权重初始化                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  原型匹配模块                          │
│  - 支持集特征聚合                      │
│  - 原型构建与匹配                      │
│  - 多粒度原型建模                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  C2F 解码器                           │
│  - 由粗到细的特征解码                  │
│  - 边界信息强化                        │
│  - 任务自适应适配                      │
└─────────────────────────────────────┘
    ↓
┌─────────┬─────────┬─────────┐
│ 分割头   │ 分类头   │ 检测头   │
│ Seg     │ Cls     │ Det     │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
  掩码      类别      边界框
```

### 核心创新

#### 1. 多粒度原型建模

```python
# 前景原型
proto_fg = MaskedAvgPool(features, mask)

# 背景原型
proto_bg = MaskedAvgPool(features, 1 - mask)

# 多尺度原型融合
proto_multi = MultiScaleFusion(proto_fg, proto_bg)
```

#### 2. 任务自适应模块

```python
# 快速适配新类别
adapter = TaskAdaptiveModule(
    support_features,
    query_features
)
adapted_features = adapter.adapt()
```

#### 3. 测试时增强 (TTA)

```python
# 多尺度测试
scales = [0.75, 1.0, 1.25]
predictions = [model(image, scale=s) for s in scales]
final_pred = aggregate(predictions)
```

---

## 实验结果

### 5.2 模型检测性能对比 (20-shot)

| 方法 | 类别 | Precision | Recall | F1 | AP50 | mIoU | 类别准确率 |
|:----:|:----:|:---------:|:------:|:--:|:----:|:----:|:----------:|
| U-Net | 分割基线 | 0.74 | 0.71 | 0.72 | 0.68 | 0.66 | 0.70 |
| SINet | COD方法 | 0.81 | 0.77 | 0.79 | 0.75 | 0.73 | 0.76 |
| UGTR | COD方法 | 0.85 | 0.81 | 0.83 | 0.79 | 0.77 | 0.80 |
| YOLOv8s | 检测方法 | 0.77 | 0.84 | 0.80 | 0.81 | 0.70 | 0.81 |
| **Ours** | **本文方法** | **0.91** | **0.93** | **0.92** | **0.90** | **0.88** | **0.92** |

### 5.3 伪装目标检测效果

| 场景 | 最佳基线 | Ours | 提升 |
|:----:|:--------:|:----:|:----:|
| 整体 | 0.82 | **0.89** | +7 |
| 颜色融合 | 0.80 | **0.88** | +8 |
| 纹理融合 | 0.79 | **0.87** | +8 |
| 小目标 | 0.79 | **0.83** | +4 |
| 遮挡/复杂背景 | 0.78 | **0.85** | +7 |

### 5.4 小样本泛化能力

| 设置 | 最佳基线 | Ours | 提升 |
|:----:|:--------:|:----:|:----:|
| 1-shot | 0.69 | **0.78** | +9 |
| 3-shot | 0.75 | **0.84** | +9 |
| 5-shot | 0.80 | **0.88** | +8 |
| Base类 | 0.85 | **0.89** | +4 |
| Novel类 | 0.75 | **0.84** | +9 |

> 📊 详细结果见 [实验表格数据.md](实验表格数据.md)

---

## 使用文档

### 训练参数说明

```bash
python train.py [OPTIONS]

必需参数:
  --mode          训练模式 [fewshot|full|ablation]
  --dataset       数据集 [COD10K|IP102|CAMO|NC4K]

可选参数:
  --shot          小样本设置 [1|3|5|20] (默认: 5)
  --epochs        训练轮数 (默认: 50)
  --batch_size    批次大小 (默认: 4)
  --lr            学习率 (默认: 1e-4)
  --output_dir    输出目录 (默认: checkpoints)
  --resume        恢复训练权重路径
```

### 推理参数说明

```bash
python inference.py [OPTIONS]

必需参数:
  --weights       模型权重路径

输入选项 (三选一):
  --image         单张图像路径
  --folder        图像文件夹路径
  --video         视频文件路径

可选参数:
  --output        输出目录 (默认: output)
  --device        推理设备 [cuda|cpu] (默认: cuda)
  --num_classes   类别数量 (默认: 102)
```

### Python API 使用

```python
from inference import PestDetector

# 初始化检测器
detector = PestDetector(
    weights_path='checkpoints/model_final.pth',
    device='cuda',
    num_classes=102
)

# 单张图像推理
results = detector.infer_single('test.jpg')

print(f"类别: {results['class_name']}")
print(f"置信度: {results['confidence']:.3f}")

# 可视化结果
detector.draw_results('test.jpg', results, 'output.jpg')
```

---

## 权重文件

### 当前可用权重

| 文件名 | 描述 | 训练设置 | F1 |
|:------:|:----:|:--------:|:--:|
| demo1_improved_model.pth | 示例权重 | 5-shot, COD10K | 0.88 |

### 生成测试权重

```bash
# 生成单个权重
python generate_weights.py --output checkpoints/model_final.pth

# 生成所有权重
python generate_weights.py --all
```

> ⚠️ 生成的权重是随机初始化的，仅用于测试代码流程

---

## 系统性能

### 测试环境

| 项目 | 配置 |
|:----:|:----:|
| 操作系统 | Windows 11 |
| CPU | Intel Core i7-12700H |
| GPU | NVIDIA GeForce RTX 3060 (6GB) |
| 内存 | 16GB DDR5 |

### 性能指标

| 指标 | 数值 |
|:----:|:----:|
| 推理速度 | 28.4 ms/帧 (35 fps) |
| GPU显存占用 | 1420 MB |
| CPU占用率 | 28% |
| 内存占用 | 1.2 GB |
| 前端帧率 | 62 fps |
| 全链路耗时 | 20.2秒 (210帧) |

---

## 常见问题

### Q1: 如何处理权重丢失？

**A:** 使用以下方法之一：
1. 重新训练：`python train.py --mode fewshot --shot 5`
2. 生成测试权重：`python generate_weights.py --output checkpoints/model_final.pth`

### Q2: CUDA内存不足怎么办？

**A:** 尝试以下方法：
```bash
# 减小batch size
python train.py --batch_size 2

# 使用CPU训练
python train.py --device cpu
```

### Q3: 如何添加新的害虫类别？

**A:** 
1. 准备新类别的样本图像
2. 更新 `num_classes` 参数
3. 使用小样本训练模式微调

### Q4: 支持哪些图像格式？

**A:** 支持常见格式：JPG, JPEG, PNG, BMP, TIFF

### Q5: 如何在自己的数据集上训练？

**A:** 
1. 按照数据集格式组织数据
2. 修改 `configs/datasets.yaml` 配置
3. 运行训练脚本

---

## 引用

如果您使用了本项目的代码，请引用：

```bibtex
@misc{pest_detection_2024,
  title={隐害先知: 小样本驱动的农林虫害智检系统},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/pest-detection}}
}
```

---

## 许可证

本项目仅供学术研究使用，请勿用于商业用途。

---

## 致谢

- [COD10K](https://xueliancheng.github.io/COD-project/) - 伪装目标检测数据集
- [IP102](https://github.com/xpwu95/IP102) - 农林害虫数据集
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

<div align="center">

**如有问题，请提交Issue或联系作者**

Made with ❤️ for Agricultural Pest Detection

</div>
