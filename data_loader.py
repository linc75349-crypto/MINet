from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset


# ========================== Rescale ==========================
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        img = transform.resize(
            image,
            (self.output_size, self.output_size),
            mode='constant'
        )

        lbl = transform.resize(
            label,
            (self.output_size, self.output_size),
            mode='constant',
            order=0,
            preserve_range=True
        )

        return {'image': img, 'label': lbl, 'name': name}


# ========================== RandomCrop ==========================
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label, 'name': name}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return {'image': image, 'label': label, 'name': name}


# ========================== ToTensor（已修复） ==========================
class ToTensor(object):
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        # -------- 处理 label --------
        if np.max(label) > 0:
            label = label / np.max(label)

        # 保证 label 是 H×W×1
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]

        # -------- 处理 image --------
        image = image / np.max(image)

        # 灰度图转3通道
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        tmpImg = np.zeros_like(image)

        tmpImg[:, :, 0] = (image[:, :, 0] - 0.4669) / 0.2437
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.4669) / 0.2437
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.4669) / 0.2437

        # HWC → CHW
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        # ✅ 关键：统一 float32（彻底解决报错）
        return {
            'image': torch.from_numpy(tmpImg).float(),
            'label': torch.from_numpy(tmpLbl).float(),
            'name': name
        }


# ========================== Dataset ==========================
class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        image = io.imread(self.image_name_list[idx])
        name = self.image_name_list[idx].split('/')[-1][:-4]

        # -------- 读取 label --------
        if len(self.label_name_list) == 0:
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        # 转单通道
        if len(label_3.shape) == 3:
            label = label_3[:, :, 0]
        else:
            label = label_3

        # -------- 维度统一 --------
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]

        sample = {'image': image, 'label': label, 'name': name}

        # -------- 数据增强 --------
        if self.transform:
            sample = self.transform(sample)

        # ✅ 终极保险（防止任何漏网之鱼）
        if isinstance(sample['image'], torch.Tensor):
            sample['image'] = sample['image'].float()

        if isinstance(sample['label'], torch.Tensor):
            sample['label'] = sample['label'].float()

        return sample
