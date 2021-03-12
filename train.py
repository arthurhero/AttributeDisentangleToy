import os
import sys 
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from opencv_utils import *
from model import ADNet
from data_utils import SUNAttribute

data_root = '../datasets/sun'
batch_size = 8
num_epochs = 10
lr = 10e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_dataset = SUNAttribute(data_root, 'train', 224, 224, freq_file='freq.txt')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
val_dataset = SUNAttribute(data_root, 'val', 224, 224, freq_file='freq.txt')
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

def train_model(model, dataloaders, criterion, optimizer, ckpt_path, best_ckpt_path, num_epochs):
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print("Loaded ckpt!")

    best_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            step = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            dataloader = dataloaders[phase]
            
            '''
                '''
            if phase == 'train':
                total = len(dataloader.dataset.imgs)
                pos_freq = dataloader.dataset.freq
                neg_freq = total - pos_freq
                pos_weights = ((1/pos_freq).pow(1/3)).to(device)
                neg_weights = ((1/neg_freq).pow(1/3)).to(device)
                normalize = (pos_weights.pow(2)+neg_weights.pow(2)).pow(0.5)
                pos_weights = 2*pos_weights / normalize
                neg_weights = 2*neg_weights / normalize

            for i,(inputs, labels) in enumerate(dataloader):
                #print(step)
                inputs = inputs.to(device)
                labels = labels.to(device) # b x 102
                if epoch == 0:
                    l_ = (labels>0.5).float()
                    corr_labels = l_.unsqueeze(2).bmm(l_.unsqueeze(1)) # b x 102 x 102

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if epoch == 0:
                        outputs, corr_mat, conds, atr_vec = model(inputs, corr_labels)
                    else:
                        outputs, corr_mat, conds, atr_vec = model(inputs)
                    loss = criterion(corr_mat, conds)
                    if epoch > 0:
                        if phase == 'train':
                            weights = pos_weights*labels + neg_weights*(1-labels)
                            loss += -labels*outputs.log()*weights - (1-labels)*(1-outputs).log()*weights
                        else:
                            loss += criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.mean(((outputs.data>=0.5) == (labels.data>=0.5)).float())

                if step % 100 == 99:
                    step_loss = running_loss / (step+1)
                    step_acc = running_corrects.double() / (step+1)
                    print('Step {}: {:.4f} Acc: {:.4f}'.format(step, step_loss, step_acc))
                    torch.save(model.state_dict(), ckpt_path)
                if phase == 'train':
                    step += 1

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_ckpt_path)



if __name__ == '__main__':
    adnet = ADNet(3,102)
    optimizer = torch.optim.Adam(adnet.parameters(),lr=lr,betas=(0.5,0.9))
    criterion = nn.MSELoss()
    train_model(adnet, {'train':train_dataloader, 'val':val_dataloader}, 
            criterion, optimizer, 'adnet.ckpt', 'adnet_best.ckpt', num_epochs)
