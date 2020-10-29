# -*- coding: utf-8 -*-
# @Time : 2020/10/28 22:13
# @Author : CHT
# @Site : 
# @File : train_1028.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian


import torch
from Image_Caltech_DeCAF6 import get_data
import CDAN, networks
from tensorboardX import SummaryWriter

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
feature_dim = 4096
num_classes = 10
n_clusters = 4
batch_size = 64
encoder_out_dim = 512

def tensorboard_plot(domain_src, domain_tgt, domain_name):
    print('----------------{}---------------'.format(domain_name))

    root_path = r'E:\cht_project\domain_adaptation_images\decaf6'
    dataloader_src = get_data.get_src_dataloader(root_path, domain_src)
    dataloader_tgt = get_data.get_src_dataloader(root_path, domain_tgt)

    train_epochs = 81

    classifier = networks.Classifier(in_dim=feature_dim, out_dim=num_classes).cuda()
    discriminator = networks.Discriminator(in_dim=num_classes*encoder_out_dim).cuda()

    with SummaryWriter('./runs/{}_1028'.format(domain_name)) as writer:
        CDAN.train(dataloader_src, dataloader_tgt, discriminator=discriminator, classifier=classifier, train_epochs=train_epochs,
                   writer=writer)


if __name__ == '__main__':
    from torch.backends import cudnn

    torch.backends.cudnn.benchmark = True

    dslr_path = 'dslr_decaf'
    caltech_path = 'caltech_decaf'
    amazon_path = 'amazon_decaf'
    webcam_path = 'webcam_decaf'

    tensorboard_plot(caltech_path, amazon_path, 'C_A')
    # tensorboard_plot(caltech_path, dslr_path, 'C_D')
    # tensorboard_plot(caltech_path, webcam_path, 'C_W')
    # tensorboard_plot(amazon_path, caltech_path, 'A_C')
    # tensorboard_plot(amazon_path, webcam_path, 'A_W')
    # tensorboard_plot(amazon_path, dslr_path, 'A_D')
    # tensorboard_plot(dslr_path, amazon_path, 'D_A')
    # tensorboard_plot(dslr_path, webcam_path, 'D_W')
    # tensorboard_plot(dslr_path, caltech_path, 'D_C')
    # tensorboard_plot(webcam_path, amazon_path, 'W_A')
    # tensorboard_plot(webcam_path, caltech_path, 'W_C')
    # tensorboard_plot(webcam_path, dslr_path, 'W_D')
