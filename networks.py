# -*- coding: utf-8 -*-
# @Time : 2020/10/28 20:04
# @Author : CHT
# @Site : 
# @File : networks.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian

import torch
from torch import nn
import numpy as np

encoder_out_dim = 512

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class Classifier(nn.Module):
    def __init__(self, in_dim ,out_dim):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, encoder_out_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(encoder_out_dim, out_dim)

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.reshape(x.shape[0], -1)
        x = self.encoder(x)
        y = self.fc(x)
        return x, y


class Discriminator(nn.Module):
    # 这里是特征和向量输出做内积
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        # 一些没用的参数
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))  # 保存梯度
        # 明显这里不影响adversarialNetwork, 只影响前面的classifier/encoder
        x = self.net(x)
        return x

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
