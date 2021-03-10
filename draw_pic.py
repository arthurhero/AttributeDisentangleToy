import os
import sys 
import time
import copy
import numpy as np
import scipy.io

import matplotlib as mpl
from matplotlib import pyplot

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencv_utils import *
from model import ADNet
from data_utils import SUNAttribute
from torch.utils.data import Dataset, DataLoader

# load attribute names
mat = scipy.io.loadmat(os.path.join('../datasets/sun','SUNAttributeDB','attributes.mat'))
mat = np.asarray(mat['attributes'])
atr_names = [m[0][0] for m in mat]

# load learned parameters
adnet = ADNet(3,102)
adnet.load_state_dict(torch.load('adnet_weight.ckpt', map_location=torch.device('cpu')))
'''
gt_corr = adnet.running_corr / adnet.running_corr.diagonal().unsqueeze(1)
learned_corr = torch.sigmoid(adnet.atr_vec.transpose(0,1).matmul(adnet.cond_mat).matmul(adnet.atr_vec))

# draw grids
gt_corr = gt_corr.cpu().detach().squeeze().numpy()
learned_corr = learned_corr.cpu().detach().squeeze().numpy()

img = pyplot.imshow(learned_corr,
                    cmap = 'magma')
pyplot.colorbar(img)
pyplot.xticks(ticks=list(range(len(atr_names))), labels=atr_names, rotation=90, Fontsize=3)
pyplot.yticks(ticks=list(range(len(atr_names))), labels=atr_names, Fontsize=3)
pyplot.title('Learned conditional correlation of attributes')
pyplot.tight_layout()
pyplot.savefig('learned_corr.pdf')
pyplot.show()
'''

# browse results
dataset = SUNAttribute('../datasets/sun', 'val', 224, 224)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
for i, (img, label) in enumerate(dataloader):
    pred, _,_ = adnet(img)
    for i in pred[0].topk(5)[1]:
        print(atr_names[i])
    print()
    display_torch_img(img[0],False)
