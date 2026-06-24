from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset


# ========================== Rescale ==========================
class Rescale(object):
    """Resize image and label to the given output size.

    Image uses bilinear interpolation; label uses nearest-neighbor
    (order=0) to preserve the binarized mask values.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        img = transform.resize(
            image, (self.output_size, self.output_size), mode='constant',
        )
        lbl = transform.resize(
            label, (self.output_size, self.output_size),
            mode='constant', order=0, preserve_range=True,
        )

        return {'image': img, 'label': lbl, 'name': name}


# ========================== RandomCrop ==========================
class RandomCrop(object):
    """Randomly crop a region of output_size from both image and label."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label, 'name': name}


# ========================== RandomHorizontalFlip ==========================
class RandomHorizontalFlip(object):
    """Horizontally flip both image and label with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return {'image': image, 'label': label, 'name': name}


# ========================== ToTensor ==========================
class ToTensor(object):
    """Convert numpy arrays to PyTorch tensors with Z-score normalization.

    Normalizes image pixels to [0,1], then applies Z-score standardization
    using dataset-level mean (0.4669) and std (0.2437). Converts HWC to CHW.
    """
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        # -------- process label --------
        if np.max(label) > 0:
            label = label / np.max(label)

        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]

        # -------- process image --------
        image = image / np.max(image)

        # convert grayscale to 3-channel
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # Z-score normalization (dataset-level statistics)
        tmpImg = np.zeros_like(image)
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.4669) / 0.2437
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.4669) / 0.2437
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.4669) / 0.2437

        # HWC -> CHW
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {
            'image': torch.from_numpy(tmpImg).float(),
            'label': torch.from_numpy(tmpLbl).float(),
            'name': name,
        }


# ========================== Dataset ==========================
class SalObjDataset(Dataset):
    """Salient Object Detection Dataset.

    Reads images and labels from file lists, applies optional transforms.

    Args:
        img_name_list: list of image file paths
        lbl_name_list: list of label file paths (empty for inference)
        transform: optional transform pipeline
    """
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        name = self.image_name_list[idx].split('/')[-1][:-4]

        # read label
        if len(self.label_name_list) == 0:
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        if len(label_3.shape) == 3:
            label = label_3[:, :, 0]
        else:
            label = label_3

        # unify dimensions to HWC
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]

        sample = {'image': image, 'label': label, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        # ensure float32 dtype
        if isinstance(sample['image'], torch.Tensor):
            sample['image'] = sample['image'].float()
        if isinstance(sample['label'], torch.Tensor):
            sample['label'] = sample['label'].float()

        return sample
