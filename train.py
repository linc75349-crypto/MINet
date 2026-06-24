import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import *
from model import MINet
import pytorch_ssim


# ------- loss functions -------
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)


def hybrid_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    return bce_out + ssim_out


def multi_loss_function(d1, d2, d3, d4, d5, labels):
    loss1 = hybrid_loss(d1, labels)
    loss2 = hybrid_loss(d2, labels)
    loss3 = hybrid_loss(d3, labels)
    loss4 = hybrid_loss(d4, labels)
    loss5 = hybrid_loss(d5, labels)

    return loss1 + loss2 + loss3 + loss4 + loss5


# ------- main training function -------
def main():

    # ------- paths -------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "Dataset/SD-saliency-900/")
    image_train_dir = os.path.join(dataset_dir, "Img_train/")
    label_train_dir = os.path.join(dataset_dir, "GT_train/")
    model_dir = os.path.join(base_dir, "model_save/")

    os.makedirs(model_dir, exist_ok=True)

    image_ext = '.bmp'
    label_ext = '.png'

    # ------- hyperparameters -------
    epoch_num = 80
    batch_size_train = 32

    # ------- dataset -------
    image_train_name_list = glob.glob(image_train_dir + '*' + image_ext)

    label_train_name_list = []
    for image_path in image_train_name_list:
        image_name = os.path.basename(image_path)
        imidx = os.path.splitext(image_name)[0]
        label_train_name_list.append(label_train_dir + imidx + label_ext)

    print("---")
    print("train images:", len(image_train_name_list))
    print("train labels:", len(label_train_name_list))
    print("---")

    train_num = len(image_train_name_list)

    # 修复：添加 RandomHorizontalFlip 数据增强
    salobj_dataset = SalObjDataset(
        img_name_list=image_train_name_list,
        lbl_name_list=label_train_name_list,
        transform=transforms.Compose([
            Rescale(368),
            RandomCrop(336),
            RandomHorizontalFlip(p=0.5),  # 新增：水平翻转增强
            ToTensor(flag=0)
        ])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # ------- device -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- model -------
    net = MINet().to(device)

    # ------- optimizer -------
    print("---define optimizer...")
    optimizer = optim.Adam(
        net.parameters(),
        lr=4e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0
    )

    # 新增：学习率调度器，每30个epoch衰减为原来的0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ------- training -------
    print("---start training...")
    ite_num = 0
    best_loss = float('inf')  # 用于保存最佳模型

    for epoch in range(epoch_num):
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1

            inputs = data['image'].float().to(device)
            labels = data['label'].float().to(device)

            optimizer.zero_grad()

            d1, d2, d3, d4, d5 = net(inputs)
            loss = multi_loss_function(d1, d2, d3, d4, d5, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # ------- save model (每1000迭代) -------
            if ite_num % 1000 == 0:
                save_path = os.path.join(
                    model_dir,
                    f"epoch_{epoch}_iter_{ite_num}.pth"
                )
                torch.save(net.state_dict(), save_path)

            # ------- log -------
            print(
                "[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] loss: %.6f"
                % (
                    epoch + 1,
                    epoch_num,
                    (i + 1) * batch_size_train,
                    train_num,
                    ite_num,
                    loss.item()
                )
            )

        # 计算平均epoch loss
        avg_epoch_loss = epoch_loss / len(salobj_dataloader)

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                net.state_dict(),
                os.path.join(model_dir, "MINet_best.pth")
            )
            print(f"  >>> Best model saved! Loss: {best_loss:.6f}")

        # 每个epoch保存一次
        torch.save(
            net.state_dict(),
            os.path.join(model_dir, f"epoch_{epoch}.pth")
        )

        # 更新学习率
        scheduler.step()

    # 训练结束，保存最终模型
    torch.save(net.state_dict(), os.path.join(model_dir, "MINet.pth"))
    print("-------------Training Done!-------------")
    print(f"Best loss: {best_loss:.6f}")


# ------- entry point -------
if __name__ == '__main__':
    main()
