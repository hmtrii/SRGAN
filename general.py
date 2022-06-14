import logging
import glob
import os
import yaml
import torch
import numpy as np
import random
import cv2
import shutil


def init_loger(output_dir):
    save_file = os.path.join(output_dir, 'info.log')
    logging.basicConfig(filename=save_file, format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

def create_train_dir(root_output, save_dir):
    os.makedirs(root_output, exist_ok=True)
    if save_dir:
        path = os.path.join(root_output, save_dir)
    else:
        id_dir = len(glob.glob(f'{root_output}/exp_*')) + 1
        path = os.path.join(root_output, f'exp_{id_dir}')
    os.makedirs(path, exist_ok=True)
    return path

def load_configs(config_path):
    with open(config_path, 'r') as config_file:
        configs = yaml.load(config_file, Loader=yaml.SafeLoader)
    return configs

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def standard_time(second):
    hour = int(second / 3600)
    minute = int((second - hour*3600) / 60)
    second = int(second - (hour*3600) - (minute*60))
    return f'{hour}h{minute}m{second}s'
