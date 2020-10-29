# -*- coding: utf-8 -*-
# @Time : 2020/10/28 20:04
# @Author : CHT
# @Site : 
# @File : utils.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian


# -*- coding: utf-8 -*-
# @Time : 2020/10/28 14:06
# @Author : CHT
# @Site :
# @File : utils.py
# @Software: PyCharm
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian


import torch
import numpy as np

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def entropy_loss(predict, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    # 这里权重需不需要，应该影响不大。
    if class_level_weight is None:
        class_level_weight = 1.0
    if instance_level_weight is None:
        instance_level_weight = 1.0

    entropy = -predict * torch.log(predict + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / predict.shape[0]


def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (-power))


def adjust_lr(optimizer, step, initial_lr):
    lr = inverseDecayScheduler(step, initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    a = torch.tensor(np.random.randint(1, 100, 10))
    b = torch.tensor([1,0,1,0,1,0,1,0,1,0])
    print(entropy_loss(a))
