# -*- coding: utf-8 -*-
# @Time : 2020/10/28 20:06
# @Author : CHT
# @Site : 
# @File : loss.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def Entropy(inputs):
    batch_size = inputs.shape[0]
    epsilon = 1e-5
    entropy = torch.sum(-inputs * torch.log(inputs + epsilon), dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    # 本质上就是用 encoder 的features 与 softmax的features叉乘，作为映射。
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        # 这里和算欧式距离一个意思，softmax概率扩展  乘以 特征维第一维  (列数1等于行数1)
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        # 这里就是输出 映射，得到[batch_size, num_classes, feature_dim]
        ad_out = ad_net(op_out.reshape(-1, softmax_output.shape[1] * feature.shape[1]))
        # 注意得到后是已经梯度翻转的了，不过也不影响后面哈
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.reshape[-1, random_out.shape[0]])
    batch_size = softmax_output.shape[0] // 2
    # 定义domain label
    dc_target = FloatTensor(np.array([[1]] * batch_size + [[0]] * batch_size))

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))   # 这里也是翻转后的结果
        # register_hook保留中间变量的导数 这里开始使用梯度反转
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.shape[0] // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.shape[0] // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
            target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.reshape(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target))
    else:
        return nn.BCELoss()(ad_out, dc_target)




if __name__ == '__main__':
    inputs = torch.randn(10, 5)
    print(Entropy(inputs).shape)
