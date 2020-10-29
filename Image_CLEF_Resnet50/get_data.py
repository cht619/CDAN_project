# -*- coding: utf-8 -*-
# @Time : 2020/10/28 19:58
# @Author : CHT
# @Site : 
# @File : get_data.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian

import csv
import numpy as np
import os
import torch.utils.data as data
from torchvision import datasets, transforms
import torch

feature_dim = 2048  # 一共是2049列，最后一列是labels
batch_size = 64
num_classes = 12


def get_src_dataloader(root_path, domain):
    path = os.path.join(root_path, domain)

    with open(path, encoding='utf-8') as f:
        imgs_data = np.loadtxt(f, delimiter=",")
        features = imgs_data[:, :-1]
        labels = imgs_data[:, -1]

    dataloader = data.DataLoader(
        dataset=myDataset(features, labels),
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader


class myDataset(data.Dataset):
    def __init__(self, imgs_data, labels):
        self.imgs_data = imgs_data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5],[0.5])])

    def __getitem__(self, index):
        # 传入的数据已经是numpy格式
        imgs_data = self.imgs_data[index]
        # imgs_data = imgs_data[:, np.newaxis]
        # 这里不知道为什么不加多一维，会报错，说输入只能是2或者3维
        # 加多一维 却变成[1,800,1] 太奇怪了，只能reshape
        # imgs_data = self.transform(Image.fromarray(imgs_data[:, np.newaxis])).reshape(1, 4096)
        imgs_data = self.transform(imgs_data[:, np.newaxis]).reshape(1, 2048)
        return imgs_data, self.labels[index]

    def __len__(self):
        return len(self.imgs_data)


if __name__ == '__main__':
    from torch.backends import cudnn

    torch.backends.cudnn.benchmark = True

    root_path = r'E:\cht_project\domain_adaptation_images\imageCLEF_resnet50'
    domain_c = 'c_c.csv'
    domain_i = 'i_i.csv'
    domain_p = 'p_p.csv'
    domain_ci = 'c_i.csv'
    domain_cp = 'c_p.csv'
    domain_ic = 'i_c.csv'
    domain_ip = 'i_p.csv'
    domain_pc = 'p_c.csv'
    domain_pi = 'p_i.csv'

    i_dataloader_src = get_src_dataloader(root_path, domain_i)

    print('数据集大小。I:{}'.format(len(i_dataloader_src.dataset)))