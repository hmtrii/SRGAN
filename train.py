from genericpath import exists
import yaml
import tqdm
import os

import torch
from torch.utils.data import DataLoader

from datasets import TrainSetCycleGan
from model import Generator, Discriminator


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

    train_set = TrainSetCycleGan(root_dir, train_subset)
    val_set = TrainSetCycleGan(root_dir, val_subset)
    test_set = TrainSetCycleGan(root_dir, test_subset)
    
    gen_model = Generator()
    dis_model = Discriminator()
    gen_optimizer = torch.optim.Adam(gen_model.parameters())
    dis_optimizer = torch.optim.Adam(dis_model.parameters())

    train_loader = DataLoader(train_set, batch_size=batch_size)

    for epoch in tqdm(range(epochs)):
        gen_model.train()
        dis_model.train()
        for hr_image, lr_image in train_loader:
            gen_optimizer.zero_grad()
            gen_hr_image = gen_model(lr_image)
            loss = 

        