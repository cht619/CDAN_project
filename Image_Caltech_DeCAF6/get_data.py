# -*- coding: utf-8 -*-
# @Time : 2020/10/28 22:13
# @Author : CHT
# @Site : 
# @File : get_data.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian

from scipy.io import loadmat
import numpy as np
import os
import torch.utils.data as data
from torchvision import datasets, transforms

batch_size = 32
feature_dim = 4096
num_classes = 10


class myDataset(data.Dataset):
    def __init__(self, imgs_data, labels):
        self.imgs_data = imgs_data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5],[0.5])])

    def __getitem__(self, index):
        imgs_data = np.asarray(self.imgs_data[index])
        # imgs_data = imgs_data[:, np.newaxis]
        # 这里不知道为什么不加多一维，会报错，说输入只能是2或者3维
        # 加多一维 却变成[1,800,1] 太奇怪了，只能reshape
        # imgs_data = self.transform(Image.fromarray(imgs_data[:, np.newaxis])).reshape(1, 4096)
        imgs_data = self.transform(imgs_data[:, np.newaxis]).reshape(1, 4096)
        return imgs_data, self.labels[index] - 1

    def __len__(self):
        return len(self.imgs_data)


def get_src_dataloader(root_path, domain):
    data_path = os.path.join(root_path, domain)
    domain_data = loadmat(data_path)
    # dict_keys(['__header__', '__version__', '__globals__', 'feas', 'labels'])
    dataset = myDataset(np.asarray(domain_data['feas']), np.asarray(domain_data['labels']).squeeze())
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataloader


if __name__ == '__main__':
    root_path = r'E:\cht_project\domain_adaptation_images\decaf6'
    dslr_path = 'dslr_decaf'
    caltech_path = 'caltech_decaf'
    amazon_path = 'amazon_decaf'
    webcam_path = 'webcam_decaf'
    dataloader_dslr = get_src_dataloader(root_path, dslr_path)
    dataloader_amazon = get_src_dataloader(root_path, amazon_path)
    for (imgs, labels) in dataloader_dslr:
        print(imgs.shape)
        print(labels)
