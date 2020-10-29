# -*- coding: utf-8 -*-
# @Time : 2020/10/28 20:23
# @Author : CHT
# @Site : 
# @File : CDAN.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian

import torch
from torch.autograd import Variable
from torch import nn, optim
import itertools
from utils import *
import loss, networks


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def evaluate(classifier, dataloader_src, dataloader_tgt):
    # 看看两者的结果

    classifier.eval()
    acc_src = acc_tgt = 0
    for (imgs_tgt, labels_tgt) in dataloader_tgt:
        feature_tgt = Variable(imgs_tgt.type(FloatTensor).reshape(imgs_tgt.shape[0], -1))
        labels_tgt = Variable(labels_tgt.type(LongTensor))

        _, predict_prob_tgt = classifier(feature_tgt)
        pred_tgt = predict_prob_tgt.data.max(1)[1]
        acc_tgt += pred_tgt.eq(labels_tgt.data).cpu().sum()

    for (imgs_src, labels_src) in dataloader_src:

        feature_src = Variable(imgs_src.type(FloatTensor)).reshape(imgs_src.shape[0], -1)
        labels_src = Variable(labels_src.type(LongTensor))

        _, predict_prob_src = classifier(feature_src)
        pred_src = predict_prob_src.data.max(1)[1]
        acc_src += pred_src.eq(labels_src.data).cpu().sum()
    acc_src = int(acc_src) / len(dataloader_src.dataset)
    acc_tgt = int(acc_tgt) / len(dataloader_tgt.dataset)
    print("[Src Accuracy = {:2%}, Tgt Accuracy = {:2%}]".format(acc_src, acc_tgt))
    return acc_src, acc_tgt


def train(dataloader_src, dataloader_tgt, discriminator, classifier, train_epochs, writer):
    discriminator.train()
    classifier.train()

    loss_clf = nn.CrossEntropyLoss()
    # 复习一下：momentum就是上次更新的方向和这次的梯度反向一样，那么这次就加快速度；
    # weight_decay就是 L2 regularization
    optimizer = optim.SGD(itertools.chain(classifier.parameters(), discriminator.parameters()), lr=1e-3, momentum=0.9,
                          weight_decay=0.0009)
    loss_clf_ = transfer_loss = 0
    for epoch in range(train_epochs):
        for (imgs_src, labels_src), (imgs_tgt, labels_tgt) in zip(dataloader_src, dataloader_tgt):
            imgs_src = Variable(imgs_src.type(FloatTensor)).reshape(imgs_src.shape[0], -1)
            labels_src = Variable(labels_src.type(LongTensor))

            imgs_tgt = Variable(imgs_tgt.type(FloatTensor)).reshape(imgs_tgt.shape[0], -1)
            labels_tgt = Variable(labels_tgt.type(FloatTensor))

            # train source domain
            fea_src, pred_src = classifier(imgs_src)
            fea_tgt, pred_tgt = classifier(imgs_tgt)
            fea = torch.cat((fea_src, fea_tgt), 0)
            pred = torch.cat((pred_src, pred_tgt), 0)

            # 计算概率
            softmax_out = nn.Softmax(dim=1)(pred)

            # 计算熵和discriminator loss
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([fea, softmax_out], discriminator, entropy, networks.calc_coeff(epoch))

            # classifier loss
            loss_clf_ = loss_clf(pred_src, labels_src)

            with OptimizerManager([optimizer]):
                total_loss = transfer_loss  + loss_clf_
                total_loss.backward()
        if epoch % 5 == 0:
            acc_src, acc_tgt = evaluate(classifier, dataloader_src, dataloader_tgt)
            writer.add_scalar('Train/loss_c_src', loss_clf_, epoch)
            writer.add_scalar('Train/transfer_loss', transfer_loss, epoch)
            writer.add_scalar('Evaluate/Acc_src', acc_src, epoch)
            writer.add_scalar('Evaluate/Acc_tgt', acc_tgt, epoch)


