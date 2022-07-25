# 加载数据
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from torch.utils.data import DataLoader

# function模块中定义的数据集预处理操作导入
from function import data_transforms

# ----------读取数据集-------------- #

"""""""""""""""1、数据集路径"""""""""""""""
data_dir = './hot_dog_data/'  # 路径
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


"""""""""""""""2、构建神经网络的数据集"""""""""""""""
batch_size = 8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
"""
以下代码的精简：
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transform['train']
image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transform['valid']
"""
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    # print(cat_to_name)

