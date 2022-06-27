import logging
import glob
import os
import yaml
import torch
import numpy as np
import random
import cv2
import shutil
from enum import Enum


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

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

    def get_value(self):
        value = None
        if self.summary_type is Summary.NONE:
            value = None
        elif self.summary_type is Summary.AVERAGE:
            value = self.avg
        elif self.summary_type is Summary.SUM:
            value = self.sum
        elif self.summary_type is Summary.COUNT:
            value = self.sum
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return value