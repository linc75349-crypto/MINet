"""
MINet 实验配置文件

所有超参数、路径、模型配置集中管理，训练和测试脚本从此文件读取配置。
修改配置只需编辑此文件，无需修改 train.py / test.py。
"""

import os

# ========== 基础路径 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 数据集配置 ==========
dataset = dict(
    name="SD-saliency-900",               # 数据集名称
    image_dir=os.path.join(BASE_DIR, "Dataset", "SD-saliency-900", "Img_train"),
    label_dir=os.path.join(BASE_DIR, "Dataset", "SD-saliency-900", "GT_train"),
    image_ext=".bmp",                      # 输入图片扩展名
    label_ext=".png",                      # 标注图片扩展名
)

# ========== 训练超参数 ==========
train = dict(
    epoch=80,                              # 训练轮数
    batch_size=32,                         # 批次大小
    lr=4e-3,                               # 初始学习率
    betas=(0.9, 0.999),                    # Adam 优化器参数
    eps=1e-8,
    weight_decay=0,
    lr_step=30,                            # 学习率衰减步长（epoch）
    lr_gamma=0.5,                          # 学习率衰减系数
    num_workers=0,                         # DataLoader 工作进程数
    pin_memory=True,                       # 是否使用锁页内存
)

# ========== 数据增强配置 ==========
transform = dict(
    rescale_size=368,                      # 缩放尺寸
    crop_size=336,                         # 随机裁剪尺寸
    hflip_prob=0.5,                        # 水平翻转概率
    img_mean=0.4669,                       # 图像均值（Z-score 标准化）
    img_std=0.2437,                        # 图像标准差
)

# ========== 模型配置 ==========
model = dict(
    name="MINet",
    model_save_dir=os.path.join(BASE_DIR, "model_save"),
    results_dir=os.path.join(BASE_DIR, "results"),
    checkpoint=os.path.join(BASE_DIR, "model_save", "MINet_best.pth"),  # 默认加载的权重
)

# ========== SSIM 损失配置 ==========
ssim = dict(
    window_size=11,
    size_average=True,
)

# ========== 测试配置 ==========
test = dict(
    batch_size=64,
    num_workers=0,
)
