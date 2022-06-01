import os
import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import TrainSetCycleGan
from model import Generator, Discriminator
from loss import ContentLoss
from engine import train, val, test


def load_configs(config_path):
    with open(config_path, 'r') as config_file:
        configs = yaml.load(config_file, Loader=yaml.SafeLoader)
    return configs


if __name__ == '__main__':
    config_path = 'configs.yml'
    configs = load_configs(config_path)

    os.makedirs(configs['save_model'], exist_ok=True)
    root_dir = configs['root_dir']
    train_subsets = configs['train_subsets']
    val_subsets = configs['val_subsets']
    test_subset = configs['test_subset']
    device = torch.device(configs['device'])
    batch_size = configs['batch_size']
    num_workers = configs['num_workers']
    epochs = configs['epochs']
    ith_pool = configs['ith_pool']
    jth_cnv = configs['jth_cnv']
    adversarial_weight = configs['adversarial_weight']
    width_image_transform = configs['width_image']
    height_image_transform = configs['height_image']
    upscaled_factor = configs['upscaled_factor']

    train_set = TrainSetCycleGan(root_dir, train_subsets, width_image_transform, height_image_transform, upscaled_factor)
    # val_set = TrainSetCycleGan(root_dir, val_subsets, 96, 96, 4)
    # test_set = TrainSetCycleGan(root_dir, test_subset)
    
    generator = Generator()
    discriminator = Discriminator(width_image_transform, height_image_transform)
    optimizerG = torch.optim.Adam(generator.parameters())
    optimizerD = torch.optim.Adam(discriminator.parameters())
    content_criterion = ContentLoss(ith_pool, jth_cnv).to(device)
    adversarial_criterion = nn.BCELoss().to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    writer = SummaryWriter()
    for epoch in tqdm(range(epochs)):
        train(discriminator,
              generator,
              train_loader,
              optimizerD,
              optimizerG,
              adversarial_criterion,
              content_criterion,
              adversarial_weight,
              batch_size,
              device
        )