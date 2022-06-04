import os
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import TrainSetCycleGan
from model import Generator, Discriminator
from loss import ContentLoss
from engine import train_epoch, val_epoch, test

from general import init_loger, create_train_dir, load_configs


ROOT_OUTPUT = './runs'

if __name__ == '__main__':
    config_path = 'configs.yml'
    configs = load_configs(config_path)
    save_dir = configs['save_dir']
    root_dataset = configs['root_dataset']
    train_subsets = configs['train_subsets']
    val_subsets = configs['val_subsets']
    test_subsets = configs['test_subset']
    device = torch.device(configs['device'])
    batch_size = configs['batch_size']
    num_workers = configs['num_workers']
    num_epochs = configs['num_epochs']
    ith_pool = configs['ith_pool']
    jth_cnv = configs['jth_cnv']
    adversarial_weight = configs['adversarial_weight']
    width_image_transform = configs['width_image']
    height_image_transform = configs['height_image']
    upscaled_factor = configs['upscaled_factor']

    output_dir = create_train_dir(ROOT_OUTPUT, save_dir)
    LOGGER = init_loger(output_dir)
    # writer = SummaryWriter()

    train_set = TrainSetCycleGan(root_dataset, train_subsets, width_image_transform, height_image_transform, upscaled_factor)
    # val_set = TrainSetCycleGan(root_dataset, val_subsets, 96, 96, 4)
    # test_set = TrainSetCycleGan(root_dataset, test_subset)
    generator = Generator().to(device)
    discriminator = Discriminator(width_image_transform, height_image_transform).to(device)
    optimizerG = torch.optim.Adam(generator.parameters())
    optimizerD = torch.optim.Adam(discriminator.parameters())
    content_criterion = ContentLoss(ith_pool, jth_cnv).to(device)
    adversarial_criterion = nn.BCELoss().to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    for epoch in range(num_epochs):
        head = ('\n' + '%18s'*7) % ('Epoch', 'D_real_loss', 'D_fake_loss', 'D_loss', 'content_loss', 'adversarial_loss', 'G_loss')
        print(head)
        LOGGER.info(head)
        train_epoch(discriminator,
              generator,
              train_loader,
              optimizerD,
              optimizerG,
              adversarial_criterion,
              content_criterion,
              adversarial_weight,
              batch_size,
              device,
              num_epochs,
              epoch,
              LOGGER
        )