# -*- coding: utf-8 -*-
# @Time : 2020/10/28 21:15
# @Author : CHT
# @Site : 
# @File : train_1028.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian


import torch
from Image_CLEF_Resnet50 import get_data
import CDAN, networks
from tensorboardX import SummaryWriter

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
feature_dim = 2048
num_classes = 12
n_clusters = 4
batch_size = 64
encoder_out_dim = 512

def tensorboard_plot(domain_src, domain_tgt, domain_name):
    print('----------------{}---------------'.format(domain_name))

    root_path = r'E:\cht_project\domain_adaptation_images\imageCLEF_resnet50'
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
    domain_c = 'c_c.csv'
    domain_i = 'i_i.csv'
    domain_p = 'p_p.csv'
    domain_ci = 'c_i.csv'
    domain_cp = 'c_p.csv'
    domain_ic = 'i_c.csv'
    domain_ip = 'i_p.csv'
    domain_pc = 'p_c.csv'
    domain_pi = 'p_i.csv'


    tensorboard_plot(domain_c, domain_ci, 'C_I')
    # tensorboard_plot(domain_c, domain_cp, 'C_P')
    # tensorboard_plot(domain_p, domain_pc, 'P_C')
    # tensorboard_plot(domain_p, domain_pi, 'P_I')
    # tensorboard_plot(domain_i, domain_ic, 'I_C')
    # tensorboard_plot(domain_i, domain_ip, 'I_P')