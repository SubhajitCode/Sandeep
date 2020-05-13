# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:14 2020

@author: Sandeep
"""

#%matplotlib inline


from __future__ import print_function, division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


import os
from os.path import isfile
import pandas as pd
from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() 





apply_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
BatchSize = 256 # change according to system specs
#
#dataset = datasets.BDRW(root='./BDRW', train=True, download=True, transform=apply_transform)
#trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, 
#                                          shuffle=True, num_workers=1)

#class Rescale(object):
#    """Rescale the image in a sample to a given size.
#
#    Args:
#        output_size (tuple or int): Desired output size. If tuple, output is
#            matched to output_size. If int, smaller of image edges is matched
#            to output_size keeping aspect ratio the same.
#    """
#
#    def __init__(self, output_size):
#        assert isinstance(output_size, (int, tuple))
#        self.output_size = output_size
#
#    def __call__(self, sample):
#        image, label = sample['image'], sample['label']
#
#        h, w = image.shape[:2]
#        if isinstance(self.output_size, int):
#            if h > w:
#                new_h, new_w = self.output_size * h / w, self.output_size
#            else:
#                new_h, new_w = self.output_size, self.output_size * w / h
#        else:
#            new_h, new_w = self.output_size
#
#        new_h, new_w = int(new_h), int(new_w)
#
#        img = transform.resize(image, (new_h, new_w))
#
#        # h and w are swapped for landmarks because for images,
#        # x and y axes are axis 1 and 0 respectively
#        #landmarks = landmarks * [new_w / w, new_h / h]
#
#        return {'image': img, 'label': label}   
#    
#class ToTensor(object):
#    """Convert ndarrays in sample to Tensors."""
#
#    def __call__(self, sample):
#        image, label = sample['image'], sample['label']
#
#        # swap color axis because
#        # numpy image: H x W x C
#        # torch image: C X H X W
#        image = image.transpose((2, 0, 1))
#        return {'image': torch.from_numpy(image),
#                'label': torch.from_numpy(landmarks)}
#    
#    
#scale = Rescale(256)
##crop = RandomCrop(128)
#composed = transforms.Compose([Rescale(256), ToTensor])    





class BDRWdataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, trans=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_frame = pd.read_excel(csv_file, header = None, names = ["digit", "label"])
        self.root_dir = root_dir
        self.trans = trans

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.label_frame.iloc[idx, 0]+".jpg")
        #print(img_name)
        try:
            print(isfile(img_name))
            image = io.imread(img_name)
        except:
            print(img_name)
        label = torch.tensor(int(self.label_frame.iloc[idx, 1]))
        #labels = labels.astype('float').reshape(-1, 2)
        
        

        if self.trans:
            image = self.trans(image)
            
        sample = {"image": image, "label":label}
        return sample
    
#apply_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        

    
    
    
    
    
    
    
dataset = BDRWdataset(csv_file = r'C:\Users\Sandeep\Desktop\DeepLearning\data\labels.xls',
                                    root_dir= r'C:\Users\Sandeep\Desktop\DeepLearning\data',
                                    trans = None)

#train_set, test_set = torch.utils.data.random_split(dataset, [1100, 293])


train_set=dataset


#BatchSize = 256 # change according to system specs


trainLoader = torch.utils.data.DataLoader(dataset = train_set, batch_size=BatchSize,
#                                          shuffle=True) # Creating dataloader
#testLoader = torch.utils.data.DataLoader(dataset = test_set, batch_size = BatchSize,
                                         shuffle = True)
# Validation set with random rotations in the range [-90,90]



data=next(iter(trainLoader))
print(data)

print('No. of samples in train set: '+str(len(trainLoader.dataset)))


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
    
use_gpu = torch.cuda.is_available()
net = LeNet()
print(net)
if use_gpu:
    print('GPU is avaialble!')
    net = net.cuda()
    
    

criterion = nn.CrossEntropyLoss() 
learning_rate = 0.1
num_epochs = 5

train_loss = []
train_acc = []
for epoch in range(num_epochs):
    
    running_loss = 0.0 
    running_corr = 0
        
    for i,data in enumerate(trainLoader):
        inputs,labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(),labels.cuda() 
        # Initializing model gradients to zero
        print(inputs)
        net.zero_grad() 
        # Data feed-forward through the network
        outputs = net(inputs)
        # Predicted class is the one with maximum probability
        preds = torch.argmax(outputs,dim=1)
        # Finding the loss
        loss = criterion(outputs, labels)
        # Accumulating the loss for each batch
        running_loss += loss 
        # Accumulate number of correct predictions
        running_corr += torch.sum(preds==labels)    
        
    totalLoss = running_loss/(i+1)
    # Calculating gradients
    totalLoss.backward()
    # Updating the model parameters
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
        
    epoch_loss = running_loss.item()/(i+1)   #Total loss for one epoch
    epoch_acc = running_corr.item()/1393
    
    
         
    train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graph
    train_acc.append(epoch_acc) #Saving the accuracy over epochs for plotting the graph
       
        
    print('Epoch {:.0f}/{:.0f} : Training loss: {:.4f} | Training Accuracy: {:.4f}'
          .format(epoch+1,num_epochs,epoch_loss,epoch_acc*100))








