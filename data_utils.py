import os
import numpy as np
import scipy.io

from PIL import Image
from opencv_utils import *

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

data_root = '../datasets/sun'


class SUNAttribute(Dataset):
    def __init__(self, data_root, split, ih, iw):
        mat = scipy.io.loadmat(os.path.join(data_root,'SUNAttributeDB','attributeLabels_continuous.mat'))
        self.img_atr = np.asarray(mat['labels_cv'],dtype=float) # num_img x 102
        num_img = self.img_atr.shape[0]
        self.train_size = int(round(num_img*0.95))
        self.val_size = num_img-self.train_size

        mat = scipy.io.loadmat(os.path.join(data_root,'SUNAttributeDB','attributes.mat'))
        mat = np.asarray(mat['attributes'])
        self.atrs = [m[0][0] for m in mat]

        mat = scipy.io.loadmat(os.path.join(data_root,'SUNAttributeDB','images.mat'))
        mat = np.asarray(mat['images'])
        self.imgs = [m[0][0] for m in mat]
        
        self.split = split

        self.trans = transforms.Compose([
            transforms.ColorJitter(brightness=(0.8, 1.2),
                                   contrast=(0.8, 1.2),
                                   saturation=(0.8, 1.2),
                                   hue=(-0.1, 0.1)),
            transforms.RandomResizedCrop(size=(ih,iw), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.ToTensor()
        ])
        self.trans_val = trans_ = transforms.Compose([
            transforms.Resize((ih,iw)),
            transforms.ToTensor()
        ])

    def __len__(self):
        if self.split == 'train':
            return self.train_size
        else:
            return self.val_size

    def __getitem__(self, index):
        if self.split == 'val':
            index += self.train_size
            trans = self.trans_val
        else:
            trans = self.trans
        img_path = self.imgs[index]
        img = Image.open(os.path.join(data_root,'images',img_path))
        img = trans(img)
        label = torch.from_numpy(self.atrs[index])
        return img, label

if __name__ == '__main__':
    dataset = SUNAttribute(data_root, 'train', 224, 224)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, (img, label) in enumerate(dataloader):
        print(img.shape)
        print(label.shape)
        display_torch_img(img[0],False)

