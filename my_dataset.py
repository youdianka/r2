import os

import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, flag: str, transforms=None):
        super(DriveDataset, self).__init__()
        # self.flag = "train" if train else "test"
        data_root = os.path.join(root, flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.mask = [os.path.join(data_root, "mask", i.split('.')[0] + '_segmentation.png') for i in img_names]
        # img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")]
        # self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # self.mask = [os.path.join(data_root, "mask", i) for i in img_names]
        # check files
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask[idx])
        #前景变1，背景为0。如果mask是调色板格式，不需要除以255
        mask = np.array(mask) / 255
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == '__main__':
    train_dataset = DriveDataset("D:\wz\AllData\ISIC2017", flag='train')
    print(len(train_dataset))

    val_dataset = DriveDataset("D:\wz\AllData\ISIC2017", flag='val')
    print(len(val_dataset))

    test_dataset = DriveDataset("D:\wz\AllData\ISIC2017", flag='test')
    print(len(test_dataset))

    i, t = train_dataset[0]
