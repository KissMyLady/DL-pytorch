# coding:utf-8
# Author:mylady
# Datetime: 2023-07-02 01:08:44
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from d2lzh_pytorch.myUtils import Timer
from d2lzh_pytorch.myUtils import get_dataloader_workers
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder

import os



def get_hotdog(batch_size=256):
    # 指定RGB三个通道的均值和方差来将图像通道归一化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std =[0.229, 0.224, 0.225])

    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

    data_dir = "/mnt/aiguo/ai_data/Datasets_on_HHD"
    hotdog_train_path = os.path.join(data_dir, 'hotdog/train')
    hotdog_test_path = os.path.join(data_dir, 'hotdog/test')

    train_iter = DataLoader(ImageFolder(hotdog_train_path, 
                                        transform=train_augs),
                                        batch_size, 
                                        shuffle=True)
    
    test_iter = DataLoader(ImageFolder(hotdog_test_path, 
                                       transform=test_augs),
                                       batch_size)
    return train_iter, test_iter



def main():
    pass


if __name__ == "__main__":
    main()
    pass