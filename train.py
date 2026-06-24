import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from config import dataset, model, train, transform as transform_cfg, ssim as ssim_cfg
from data_loader import Rescale, RandomCrop, RandomHorizontalFlip, ToTensor, SalObjDataset
from model import MINet
import pytorch_ssim


# ------- loss functions -------
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(
    window_size=ssim_cfg["window_size"],
    size_average=ssim_cfg["size_average"],
)


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

    # ------- prepare data -------
    image_train_name_list = glob.glob(dataset["image_dir"] + '*' + dataset["image_ext"])
    label_train_name_list = [
        os.path.join(dataset["label_dir"], os.path.splitext(os.path.basename(p))[0] + dataset["label_ext"])
        for p in image_train_name_list
    ]

    print("train images:", len(image_train_name_list))
    print("train labels:", len(label_train_name_list))

    salobj_dataset = SalObjDataset(
        img_name_list=image_train_name_list,
        lbl_name_list=label_train_name_list,
        transform=transforms.Compose([
            Rescale(transform_cfg["rescale_size"]),
            RandomCrop(transform_cfg["crop_size"]),
            RandomHorizontalFlip(p=transform_cfg["hflip_prob"]),
            ToTensor(flag=0),
        ])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["num_workers"],
        pin_memory=train["pin_memory"],
    )

    # ------- device -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- model -------
    net = MINet().to(device)

    # ------- optimizer & scheduler -------
    optimizer = optim.Adam(
        net.parameters(),
        lr=train["lr"],
        betas=train["betas"],
        eps=train["eps"],
        weight_decay=train["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train["lr_step"], gamma=train["lr_gamma"])

    # ------- training -------
    os.makedirs(model["model_save_dir"], exist_ok=True)
    print("---start training...")
    ite_num = 0
    best_loss = float('inf')

    for epoch in range(train["epoch"]):
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

            if ite_num % 1000 == 0:
                torch.save(
                    net.state_dict(),
                    os.path.join(model["model_save_dir"], f"epoch_{epoch}_iter_{ite_num}.pth"),
                )

            print(
                "[epoch: %3d/%3d, batch: %5d, ite: %d] loss: %.6f"
                % (epoch + 1, train["epoch"], (i + 1) * train["batch_size"], ite_num, loss.item())
            )

        avg_epoch_loss = epoch_loss / len(salobj_dataloader)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(net.state_dict(), os.path.join(model["model_save_dir"], "MINet_best.pth"))
            print(f"  >>> Best model saved! Loss: {best_loss:.6f}")

        torch.save(net.state_dict(), os.path.join(model["model_save_dir"], f"epoch_{epoch}.pth"))
        scheduler.step()

    torch.save(net.state_dict(), os.path.join(model["model_save_dir"], "MINet.pth"))
    print("-------------Training Done!-------------")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == '__main__':
    main()
