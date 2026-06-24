import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import numpy as np
import glob

from data_loader import *

from model import MINet


def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)
    out = (x - MIN) / (MAX - MIN)
    return out


def save_output(image_dir, image_name, pred, save_dir):
    predict = pred
    predict = predict.squeeze()
    predict = predict.cpu().data.numpy()
    predict = Image.fromarray(predict * 255).convert('RGB')

    # 提取纯文件名（去掉路径）
    base_name = os.path.basename(image_name)

    # 读取原图
    image = Image.open(os.path.join(image_dir, base_name + '.bmp')).convert('RGB')
    predict = predict.resize((image.size[0], image.size[1]), resample=Image.BILINEAR)

    # 创建并列对比图
    compare = Image.new('RGB', (image.size[0] * 2, image.size[1]))
    compare.paste(image, (0, 0))
    compare.paste(predict, (image.size[0], 0))

    compare.save(os.path.join(save_dir, base_name + '_compare.png'))


if __name__ == '__main__':
    # --------- Define the address and image format ---------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "Dataset/SD-saliency-900/Img_train/")
    prediction_dir = os.path.join(base_dir, "results/")
    model_dir = os.path.join(base_dir, "model_save/MINet_best.pth")

    # 创建结果目录
    os.makedirs(prediction_dir, exist_ok=True)

    img_name_list = glob.glob(image_dir + '*.bmp')

    # --------- Load the data ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([Rescale(368), ToTensor(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=64, shuffle=False, num_workers=0)

    # --------- Define the model ---------
    print("...load MINet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MINet()
    net.load_state_dict(torch.load(model_dir, map_location=device))
    net = net.to(device)
    net.eval()

    # --------- Generate prediction images ---------
    print("开始生成预测结果...\n")

    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):
            print(f"处理进度: {i_test+1}/{len(test_salobj_dataloader)}")
            inputs_test, name_list = data_test['image'], data_test['name']
            inputs_test = inputs_test.float().to(device)

            d1, d2, d3, d4, d5 = net(inputs_test)

            for i in range(d1.shape[0]):
                pred = d1[i, 0, :, :]
                pred = normPRED(pred)
                save_output(image_dir, name_list[i], pred, prediction_dir)

            del d1, d2, d3, d4, d5

    print("\n-------------Done!-------------")
