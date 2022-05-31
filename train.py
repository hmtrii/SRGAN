from genericpath import exists
import yaml
import tqdm
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import TrainSetCycleGan
from model import Generator, Discriminator
from loss import ContentLoss, AversarialLoss


def load_configs(config_path):
    with open(config_path, 'r') as config_file:
        configs = yaml.load(config_file, Loader=yaml.SafeLoader)
    return configs


if __name__ == '__main__':
    config_path = 'configs.yml'
    configs = load_configs(config_path)

    os.makedirs(configs['save_model'], exist_ok=True)
    root_dir = configs['root_dir']
    train_subset = configs['train_subset']
    val_subset = configs['val_subset']
    test_subset = configs['test_subset']
    device = torch.device(configs['device'])
    batch_size = configs['batch_size']
    num_workers = configs['num_workers']
    epochs = configs['epochs']
    ith_pool = configs['ith_pool']
    jth_cnv = configs['jth_cnv']

    train_set = TrainSetCycleGan(root_dir, train_subset)
    val_set = TrainSetCycleGan(root_dir, val_subset)
    test_set = TrainSetCycleGan(root_dir, test_subset)
    
    gen_model = Generator()
    dis_model = Discriminator()
    gen_optimizer = torch.optim.Adam(gen_model.parameters())
    dis_optimizer = torch.optim.Adam(dis_model.parameters())
    gen_criterion = ContentLoss(ith_pool, jth_cnv).to(device)
    adver_criterion = nn.BCELoss().to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    for epoch in tqdm(range(epochs)):
        gen_model.train()
        dis_model.train()
        for hr_tensor, lr_tensor in train_loader:
            sr_tensor = gen_model(lr_tensor)
            # Discrim network
            dis_optimizer.zero_grad()
            prob_sr = dis_model(sr_tensor)
            prob_hr = dis_model(hr_tensor)
            fake_labels = torch.full([batch_size, 0], 0.0, dtype=prob_sr.dtype, device=device)
            real_labels = torch.full([batch_size, 0], 1.0, dtype=prob_hr.dtype, device=device)
            dis_loss_sr = adver_criterion(prob_sr)
            dis_loss_hr = adver_criterion(hr_tensor)
            dis_loss = (dis_loss_sr + dis_loss_hr)
            # Gen network
            gen_optimizer.zero_grad()
            content_loss = gen_criterion(hr_tensor, lr_tensor)
            
            

        