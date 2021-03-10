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
    def __init__(self, data_root, split, ih, iw, freq_file='freq.txt'):
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
                                   saturation=(0.8, 1.2)),
                                   #hue=(-0.1, 0.1)),
            transforms.RandomResizedCrop(size=(ih,iw), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.ToTensor()
        ])
        self.trans_val = trans_ = transforms.Compose([
            transforms.Resize((ih,iw)),
            transforms.ToTensor()
        ])
        with open(freq_file,'r') as f:
            self.freq = [float(l.rstrip().split()[-1]) for l in f.readlines()]
            f.close()
        self.freq = torch.Tensor(self.freq)

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
        img = Image.open(os.path.join(data_root,'images',img_path)).convert('RGB')
        img = trans(img)
        label = torch.from_numpy(self.img_atr[index]).float()
        return img, label

if __name__ == '__main__':
    dataset = SUNAttribute(data_root, 'train', 224, 224)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    freq = np.zeros(102)
    for i, (img, label) in enumerate(dataloader):
        #display_torch_img(img[0],False)
        freq += (label>0.5).float().sum(0).numpy()
    for i in range(len(freq)):
        print(dataset.atrs[i],freq[i])
    import matplotlib as mpl
    from matplotlib import pyplot

    pyplot.bar(list(range(102)), freq, width=0.8)
    pyplot.xticks(ticks=list(range(102)), labels=dataset.atrs, rotation=90, Fontsize=3)
    pyplot.title('Attribute frequency in SUN Attribute training split')
    pyplot.tight_layout()
    pyplot.savefig('freq.pdf')
    pyplot.show()
