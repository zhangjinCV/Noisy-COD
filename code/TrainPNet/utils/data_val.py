import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import albumentations as A


def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


aug = A.Compose([
    A.ColorJitter(0.5, 0.5, 0.5, 0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.GaussNoise(p=0.2),
    A.Blur(p=0.2),
    A.ShiftScaleRotate(rotate_limit=30),
    A.ToGray(p=0.2),
    A.Emboss(p=0.5),
    A.Posterize(p=0.5),
    A.Perspective(p=0.5)
], additional_targets={'mask': 'mask', 'edge': 'mask'})



# dataset for training
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, istraining=True):
        self.trainsize = trainsize
        self.istraining = istraining
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')] 
        self.gts = [i.replace("image", "mask").replace(".jpg", ".png") for i in self.images] 
        self.edges = [i.replace("mask", "edge") for i in self.gts] 

        if 'CAMO_COD_train' in image_root:
            fully_ration = len(self.images)
            weakly_ration = 4040 - fully_ration
            k = (weakly_ration / 4) / fully_ration
            k = int(k)
            self.images *= k 
            self.gts *= k
            self.edges *= k

        if self.istraining:
            self.images = np.array(sorted(self.images))
            self.gts = np.array(sorted(self.gts))
            self.edges = np.array(sorted(self.edges))
        else:
            self.images = np.array(sorted(self.images))
            self.gts = np.array(sorted(self.gts))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        name = os.path.basename(self.images[index])
        gt = self.binary_loader(self.gts[index])
        H, W = image.size
        if self.istraining:
            edge = self.binary_loader(self.edges[index])
            image, gt, edge = np.array(image).astype(np.uint8), np.array(gt).astype(np.uint8), np.array(edge).astype(np.uint8)
            augmented = aug(image=image, mask=gt, edge=edge)
            image, gt, edge = augmented['image'], augmented['mask'], augmented['edge']
            gt = np.where(gt > 0.2 * 255, 255, 0).astype(np.uint8)
            edge = np.where(edge > 0.2 * 255, 255, 0).astype(np.uint8)
            image, gt, edge = Image.fromarray(image),  Image.fromarray(gt), Image.fromarray(edge)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.istraining:
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt, [H, W], name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training and testing
def get_train_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=6, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, trainsize, istraining=True)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_test_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=0, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, trainsize, istraining=False)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    train_loader = get_train_loader(r"F:\Dataset\COD\CAMO_COD_train\image\\", r"F:\Dataset\COD\CAMO_COD_train\mask\\", 1, 352)
    for idx, (image, gt, edge) in enumerate(train_loader):
        print(image.max(), gt.shape)
        image = image.cpu().numpy().squeeze().transpose((1, 2, 0)) * 255.
        gt = gt.cpu().numpy().squeeze() * 255.
        edge = edge.cpu().numpy().squeeze() * 255.
        plt.subplot(2, 2, 1)
        plt.imshow(image.astype(np.uint8))
        plt.subplot(2, 2, 2)
        plt.imshow(gt)
        plt.subplot(2, 2, 4)
        plt.imshow(edge)
        plt.show()
        input()
