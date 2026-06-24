# MINet 说明

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.4.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

MINet：Multiscale Interactive Network for Real-Time Salient Object Detection of Strip Steel Surface Defects（**多尺度交互网络：带钢表面缺陷实时显著性目标检测**）

---

## 一、项目概述

本项目基于 MINet（Multiscale Interactive Network）架构，实现了**带钢表面缺陷的实时显著性目标检测**。核心创新是 **MI Module（多尺度交互模块）**，通过扩展深度可分离卷积（DSConv），以极低的参数量实现高精度的缺陷检测。

### 模型特点

| 指标 | 数值 |
|---|---|
| **参数量** | **0.28M**（极轻量） |
| **计算量** | 0.30G FLOPs |
| **GPU 速度** | **721 FPS**（NVIDIA GTX 2080Ti） |
| **CPU 速度** | 6.3 FPS（i9-9900X） |
| **输入尺寸** | 368 × 368 |

### 项目目录结构

| 文件/目录 | 说明 |
|---|---|
| train.py | 训练入口，包含多损失函数（BCE + SSIM）和训练循环 |
| test.py | 测试入口，加载模型权重并生成显著性预测图 |
| FPS.py | 计算模型推理速度（FPS） |
| data_loader.py | 数据加载与预处理（Rescale、RandomCrop、RandomHorizontalFlip、ToTensor） |
| model/MINet.py | 核心模型定义（MINet 网络结构） |
| model/basic.py | 基础模块（DSConv3x3、MI_Module、ConvOut 等） |
| pytorch_ssim/ | SSIM 损失函数实现（用于混合损失训练） |
| Dataset/ | 数据集目录（SD-saliency-900） |
| model_save/ | 训练好的模型权重保存位置 |
| results/ | 测试结果（原图+预测图对比） |
| figures/ | 论文中的架构图与效果对比图 |

---

## 二、环境搭建

### 2.1 硬件要求

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA GPU（建议 4GB+ 显存） |
| CUDA | CUDA 10.0+ |
| 系统 | Windows / Linux / macOS |

### 2.2 创建 Conda 环境

```bash
conda create -n minet python=3.7 -y
conda activate minet
```

### 2.3 安装 PyTorch

使用 conda 安装 CUDA 版 PyTorch（确保 CUDA 可用）：

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
```

如果使用更新的 PyTorch 版本，也可以兼容：

```bash
# PyTorch 1.10+ 示例
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
```

### 2.4 安装其他依赖

```bash
pip install numpy==1.18.1 scikit-image tqdm
```

---

## 三、数据集准备

### 3.1 数据集结构

本项目使用 **SD-Saliency-900** 数据集，包含带钢表面缺陷图像及其对应的显著性标签图（GT）。

数据集目录结构如下：

```
Dataset/SD-saliency-900/
├── Img_train/      # 训练图像（.bmp 格式）
└── GT_train/       # 训练标签（.png 格式）
```

### 3.2 数据集说明

数据集包含三种类型的钢板表面缺陷：
- **Inclusion（夹杂）**
- **Patches（斑块）**
- **Scratches（划痕）**

### 3.3 数据集准备

将数据集放置在 `Dataset/SD-saliency-900/` 目录下，确保图像为 `.bmp` 格式，标签为 `.png` 格式。数据加载器会自动匹配对应的图像和标签文件名。

---

## 四、训练与测试

### 4.1 激活环境

```bash
conda activate minet
cd MINet
```

### 4.2 开始训练

默认训练 80 个 epoch，batch size 为 32：

```bash
python train.py
```

训练过程中会自动保存：
- **最佳模型**：`model_save/MINet_best.pth`（基于验证损失）
- **每个 epoch 的模型**：`model_save/epoch_{epoch}.pth`
- **每 1000 次迭代的模型**：`model_save/epoch_{epoch}_iter_{ite}.pth`

训练完成后保存最终模型：`model_save/MINet.pth`

### 4.3 训练输出解读

训练过程中每步输出格式如下：

```
[epoch:  1/80, batch:   32/900, ite: 1] loss: 0.853216
[epoch:  1/80, batch:   64/900, ite: 2] loss: 0.792145
...
  >>> Best model saved! Loss: 0.124356
```

| 指标 | 说明 |
|---|---|
| epoch | 当前训练轮次 / 总轮次 |
| batch | 已处理的样本数 / 总样本数 |
| ite | 总迭代次数 |
| loss | 混合损失（BCE + SSIM），期望值 0.1~1.5 |

### 4.4 运行测试

训练完成后，使用测试脚本生成显著性预测图：

```bash
python test.py
```

测试脚本将：
1. 加载 `model_save/MINet_best.pth` 最佳模型权重
2. 对 `Dataset/SD-saliency-900/Img_train/` 中的图像进行预测
3. 在 `results/` 目录下生成原图与预测图的左右对比图（`*_compare.png`）

### 4.5 计算推理速度

```bash
python FPS.py
```

---

## 五、模型结构

### 5.1 MI Module（多尺度交互模块）

![MI Module](figures/MI.png)

MI Module 是 MINet 的核心组件，其工作原理：

1. **多尺度特征提取**：使用 4 个不同膨胀率的深度卷积（DWConv）并行提取多尺度特征
2. **特征交互融合**：通过 Channel Shuffle 操作实现跨尺度特征交互
3. **逐点卷积**：每个通道独立使用 PWConv 进行特征选择
4. **残差连接**：融合后的特征与输入相加，保留原始信息

### 5.2 整体架构

MINet 采用 Encoder-Decoder 结构：

- **Encoder**：5 个 stage，基于 MI Module 构建的轻量级实时骨干网络
- **Decoder**：5 个 stage，使用不同膨胀率的 DSConv，逐步恢复空间分辨率
- **侧边输出**：每层 decoder 均有显著性预测输出，共同参与训练

### 5.3 损失函数

采用混合损失函数（Hybrid Loss）：
- **BCE Loss**：二值交叉熵损失，衡量像素级分类精度
- **SSIM Loss**：结构相似性损失，保持预测图的局部结构一致性
- 多侧输出共同计算损失，监督各层特征学习

---

## 六、定量与定性结果

### 6.1 定量对比

![Quantitative Comparison](figures/quan.png)

### 6.2 定性对比

![Qualitative Comparison](figures/qual.png)

---

## 七、常见问题与解决方案

**Q1: ModuleNotFoundError: No module named 'xxx'**

当前环境未安装对应包，执行 `pip install xxx` 安装。

**Q2: FileNotFoundError 找不到数据集**

检查 `Dataset/SD-saliency-900/` 目录结构是否正确，确保图像与标签文件名一一对应。

**Q3: CUDA out of memory**

显存不足。减小 `train.py` 中 `batch_size_train`（如 32 → 8 或 16）。

**Q4: 如何确认正在使用 GPU？**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
输出 `True` 表示使用 GPU，`False` 表示使用 CPU。

**Q5: 加载模型时 state_dict 不匹配**

如果用不同配置或不同版本的代码训练，加载时可能遇到 key 不匹配。可设置 `strict=False`：
```python
net.load_state_dict(torch.load('model.pth'), strict=False)
```

---

## 八、引用

如果您在研究中使用了 MINet，请引用原始论文：

```bibtex
@article{shen2024minet,
  title={MINet: Multiscale Interactive Network for Real-Time Salient Object Detection of Strip Steel Surface Defects},
  author={Shen, Kunye and Zhou, Xiaofei and Liu, Zhi},
  journal={IEEE Transactions on Industrial Informatics},
  year={2024}
}
```

---

## 九、参考

- 原始论文：[MINet: Multiscale Interactive Network ...](https://arxiv.org/abs/2405.16096) (IEEE TII 2024)
- 原作者：[Kunye Shen](https://github.com/Kunye-Shen)、[Xiaofei Zhou](https://scholar.google.com.hk/citations?user=2PUAFW8AAAAJ&hl=zh-CN)、[Zhi Liu](https://scholar.google.com.hk/citations?user=Sd5VB2cAAAAJ&hl=zh-CN)
- 本仓库维护：[linc75349-crypto](https://github.com/linc75349-crypto/MINet)

如有问题，请在 GitHub 上提出 Issue。
