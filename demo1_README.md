# demo1.py 使用说明

## 概述
`demo1.py` 是改进版小样本伪装目标分割训练脚本,集成了8项核心改进技术。

## 8项改进

### 1. ✓ 注意力机制: CBAM + SE + ECA 三重注意力
- **SE (Squeeze-and-Excitation)**: 全局通道依赖建模
- **ECA (Efficient Channel Attention)**: 高效局部跨通道交互
- **CBAM**: 通道+空间联合注意力
- **融合方式**: 并行计算三种注意力,可学习权重融合 + 残差连接
- **应用位置**: FPNDecoder的每个特征层级(p2/p3/p4)

### 2. ✓ 多尺度融合: ASPP + FPN
- **ASPP (Atrous Spatial Pyramid Pooling)**: 
  - 空洞率[6, 12, 18]的多尺度卷积
  - 全局平均池化分支(GroupNorm避免batch size=1问题)
  - 应用于c4深层特征
- **FPN (Feature Pyramid Network)**:
  - 自顶向下特征融合
  - 横向连接统一通道数
  - 保留高分辨率细节

### 3. ✓ 数据增强: MixUp, CutMix, 几何变换, 颜色抖动
- **MixUp**: λ*img1 + (1-λ)*img2, β分布采样λ
- **CutMix**: 随机裁剪patch覆盖,按面积混合标签
- **几何变换**: 随机翻转(50%), 旋转(±10°)
- **颜色抖动**: 亮度/对比度/饱和度(±0.2), 色相(±0.1)
- **应用范围**: 仅在support set内混合,不跨episode

### 4. ✓ 损失函数: Focal Loss + Tversky Loss + Boundary Loss
- **Focal Loss**: γ=2.0, α=0.75, 关注难分类样本
- **Tversky Loss**: α=0.7(假阴), β=0.3(假阳), 优化不平衡分割
- **Boundary Loss**: 形态学边界检测(kernel=7), BCE对齐
- **权重配置**: [1.0, 1.0, 0.5]

### 5. ✓ 深度监督
- **多尺度损失**: log2/log3/log4/log5分别计算
- **递减权重**: {log2: 1.0, log3: 0.5, log4: 0.25, log5: 0.125}
- **标签平滑**: ε=0.1, 硬标签{0,1}→软标签{ε, 1-ε}

### 6. ✓ 测试时增强 (TTA)
- **多尺度推理**: [0.9, 1.0, 1.1]三个尺度
- **水平翻转**: 原始 + 翻转预测
- **融合策略**: 所有预测结果平均(共6个预测/样本)
- **启用方式**: `--use-tta`参数

### 7. ✓ 标签平滑
- **平滑系数**: ε=0.1
- **转换公式**: smooth_label = label * (1-ε) + (1-label) * ε
- **应用位置**: 损失计算前对target_mask处理

### 8. ✓ 学习率调度: CosineAnnealingWarmRestarts
- **初始周期**: T_0=10 epochs
- **周期倍增**: T_mult=2 (10→20→40...)
- **最小学习率**: eta_min=1e-6
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-4)

## 使用方法

### 基础训练
```bash
python demo1.py --epochs 30 --steps 200 --lr 1e-4
```

### 启用TTA评估
```bash
python demo1.py --epochs 30 --steps 200 --use-tta
```

### 仅评估(使用已有模型)
```bash
python demo1.py --eval-only --model-path checkpoints/demo1_improved_model.pth --use-tta
```

### 快速测试
```bash
python demo1.py --epochs 2 --steps 10
```

## 文件结构

### 新增文件
- `my_models/attention_modules.py`: 注意力模块(SE/ECA/CBAM/TripleAttention)
- `demo1.py`: 主训练脚本
- `checkpoints/demo1_improved_model.pth`: 训练保存的模型

### 修改文件
- `my_models/encoder_decoder.py`: 
  - 添加ASPPModule类
  - 添加FPNDecoder类(替代C2FDecoder)
  - 集成TripleAttention
- `my_models/fewshot_camouflage_seg.py`:
  - 支持use_fpn和use_attention开关
  - 默认启用FPN和三重注意力

## 运行结果示例

```
============================================================
改进版Few-shot Camouflage Segmentation训练 (demo1)
============================================================
8项改进:
1. ✓ 三重注意力 (CBAM + SE + ECA)
2. ✓ 多尺度融合 (ASPP + FPN)
3. ✓ 数据增强 (MixUp + CutMix + 几何 + 颜色)
4. ✓ 组合损失 (Focal + Tversky + Boundary)
5. ✓ 深度监督 (多尺度损失)
6. ✓ 测试时增强 (TTA)
7. ✓ 标签平滑 (epsilon=0.1)
8. ✓ 学习率调度 (CosineAnnealingWarmRestarts)
============================================================

Epoch [ 1/30] LR: 0.000100 | Loss: 1.7416 | Val Dice: 0.3337 | mIoU: 0.3339 | S: 0.4030
  -> 新最佳! Dice=0.3337

Novel测试结果:
============================================================
主指标:
  mIoU:      0.2868
  Dice:      0.3599
  S-measure: 0.4022
辅指标:
  maxF:      0.2970
  MAE:       0.5011
  PixelAcc:  0.4534
============================================================
```

## 性能预期

经过完整训练(30 epochs)后预期指标提升:
- **mIoU**: +3-5% (相比基线)
- **Dice**: +2-4%
- **边界质量**: 显著改善(Boundary Loss + TTA)
- **泛化能力**: Novel场景表现提升(MixUp/CutMix + 标签平滑)

## 注意事项

1. **显存占用**: ASPP+FPN+注意力增加约30%显存,如遇OOM可减小IMG_SIZE或batch_size
2. **训练时间**: TTA评估使推理时间增加约6倍,建议训练时禁用(--use-tta仅在最终评估启用)
3. **数据路径**: 自动处理txt文件中的绝对路径替换,无需手动修改
4. **早停机制**: patience=10,连续10轮无改进自动停止
5. **模型保存**: 最佳模型自动保存至`checkpoints/demo1_improved_model.pth`

## 消融实验

可通过修改`create_fscamo_model`参数进行消融实验:

```python
# 禁用FPN,使用原始C2FDecoder
model = create_fscamo_model(use_fpn=False, use_attention=True)

# 禁用三重注意力
model = create_fscamo_model(use_fpn=True, use_attention=False)

# 两者都禁用(基线)
model = create_fscamo_model(use_fpn=False, use_attention=False)
```

## 故障排除

### 问题1: FileNotFoundError
**原因**: 数据路径不正确  
**解决**: 检查`DATA_ROOT`是否指向正确的FSL-COD数据集目录

### 问题2: CUDA out of memory
**原因**: 显存不足  
**解决**: 
- 减小IMG_SIZE (512→448或384)
- 减少ASPP的dilation数量
- 禁用部分注意力模块

### 问题3: 训练loss不下降
**原因**: 学习率过大或损失权重不合适  
**解决**:
- 降低初始学习率 (1e-4 → 5e-5)
- 调整损失权重 (focal/tversky/boundary)
- 检查数据加载是否正确

## 技术支持

如有问题,请检查:
1. PyTorch版本 >= 1.10
2. CUDA可用且驱动正常
3. 数据集完整且路径正确
4. 依赖包已安装(torchvision, PIL, numpy)
