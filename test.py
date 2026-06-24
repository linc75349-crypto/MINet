import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from config import dataset, model, test as test_cfg, transform as transform_cfg
from data_loader import Rescale, ToTensor, SalObjDataset
from model import MINet


def normPRED(x):
    """Min-Max 归一化到 [0, 1]"""
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def save_output(image_dir, image_name, pred, save_dir, image_ext=".bmp"):
    """生成原图与预测图的左右对比图"""
    predict = pred.squeeze().cpu().data.numpy()
    predict = Image.fromarray((predict * 255).astype("uint8")).convert("RGB")

    base_name = os.path.basename(image_name)
    image = Image.open(os.path.join(image_dir, base_name + image_ext)).convert("RGB")
    predict = predict.resize((image.size[0], image.size[1]), resample=Image.BILINEAR)

    compare = Image.new("RGB", (image.size[0] * 2, image.size[1]))
    compare.paste(image, (0, 0))
    compare.paste(predict, (image.size[0], 0))
    compare.save(os.path.join(save_dir, base_name + "_compare.png"))


if __name__ == "__main__":
    os.makedirs(model["results_dir"], exist_ok=True)

    img_name_list = glob.glob(dataset["image_dir"] + "*" + dataset["image_ext"])

    test_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([
            Rescale(transform_cfg["rescale_size"]),
            ToTensor(flag=0),
        ]),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_cfg["batch_size"],
        shuffle=False,
        num_workers=test_cfg["num_workers"],
    )

    print("...load MINet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MINet()
    net.load_state_dict(torch.load(model["checkpoint"], map_location=device, weights_only=True))
    net = net.to(device)
    net.eval()

    print("Generating predictions...\n")
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            print(f"Batch {i + 1}/{len(test_dataloader)}")
            inputs = data["image"].float().to(device)
            names = data["name"]

            d1, *_ = net(inputs)

            for j in range(d1.shape[0]):
                pred = normPRED(d1[j, 0, :, :])
                save_output(dataset["image_dir"], names[j], pred, model["results_dir"], dataset["image_ext"])

    print("\n-------------Done!-------------")
