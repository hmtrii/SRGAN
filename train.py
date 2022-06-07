import os
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda import amp

from datasets import TrainSetCycleGan
from model import Generator, Discriminator
from losses import ContentLoss
from engine import train_epoch, val_epoch, test

from general import init_loger, create_train_dir, load_configs
from metrics import PSNR, SSIM


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
    val_set = TrainSetCycleGan(root_dataset, val_subsets, width_image_transform, height_image_transform, upscaled_factor)
    # test_set = TrainSetCycleGan(root_dataset, test_subset)
    
    generator = Generator().to(device)
    discriminator = Discriminator(width_image_transform, height_image_transform).to(device)
    
    optimizerG = torch.optim.Adam(generator.parameters())
    optimizerD = torch.optim.Adam(discriminator.parameters())

    content_criterion = ContentLoss(ith_pool, jth_cnv).to(device)
    adversarial_criterion = nn.BCELoss().to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    psnr_metric = PSNR().to(device)
    ssim_metric = SSIM()

    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        show_train = ('\n' + '%18s'*7) % ('Epoch', 'D_real_loss', 'D_fake_loss', 'D_loss', 'content_loss', 'adversarial_loss', 'G_loss')
        print(show_train)
        LOGGER.info(show_train)
        train_epoch(discriminator,
              generator,
              train_loader,
              optimizerD,
              optimizerG,
              adversarial_criterion,
              content_criterion,
              adversarial_weight,
              scaler,
              batch_size,
              device,
              num_epochs,
              epoch,
              LOGGER
        )

        show_val = ('%18s'*7) % ('', '', '', '', '', 'PSNR', 'SSIM')
        print(show_val)
        val_epoch(generator,
                  val_loader,
                  psnr_metric,
                  ssim_metric,
                  batch_size,
                  LOGGER)