import time
import logging
from tqdm import tqdm
from enum import Enum
from typing import List


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda import amp


def train_epoch(discriminator: nn.Module,
          generator: nn.Module,
          dataloader: DataLoader,
          optimizerD: optim.Optimizer,
          optimizerG: optim.Optimizer,
          adversarial_criterion: nn.BCELoss,
          content_criterion: nn.Module,
          adversarial_weight: float,
          scaler: amp.GradScaler, 
          batch_size: int,
          device: torch.device,
          num_epochs: int,
          epoch: int,
          LOGGER: logging.Logger,
        ) -> None:
    dis_fake_losses = AverageMeter('Discriminator loss for real samples', ':6.6f')
    dis_real_losses = AverageMeter('Discriminator loss for fake samples', ':6.6f')
    dis_losses = AverageMeter('Dicriminator loss', ':6.6f')
    content_losses = AverageMeter('Content loss', ':6.6f')
    adversarial_losses = AverageMeter('Adversarial loss', ':6.6f')
    gen_losses = AverageMeter('Generator loss', ':6.6f')
    discriminator.train()
    generator.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (hr_images, lr_images) in pbar:
        ### Discriminator network
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        optimizerD.zero_grad()
        
        # Create lable for classification task
        real_labels = torch.full([batch_size, 1], 1.0, dtype=hr_images.dtype, device=device)
        fake_labels = torch.full([batch_size, 1], 0.0, dtype=lr_images.dtype, device=device)
        
        # Real samples
        with amp.autocast():
            prob_hr = discriminator(hr_images)
            dis_real_loss = adversarial_criterion(prob_hr, real_labels)
        scaler.scale(dis_real_loss).backward()

        # Generate sample
        with amp.autocast():
            sr_images = generator(lr_images)
            prob_sr = discriminator(sr_images.detach().clone())
            dis_fake_loss = adversarial_criterion(prob_sr, fake_labels)
            dis_loss = (dis_fake_loss + dis_real_loss)
        scaler.scale(dis_fake_loss).backward()
        scaler.step(optimizerD)
        scaler.update()

        ### Generator network
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        optimizerG.zero_grad()
        with amp.autocast():
            content_loss = content_criterion(hr_images, sr_images)
            adversarial_loss = adversarial_criterion(discriminator(sr_images), real_labels)
            gen_loss = content_loss + adversarial_weight*adversarial_loss
        scaler.scale(gen_loss).backward()
        scaler.step(optimizerG)
        scaler.update()

        ### Track values
        dis_fake_losses.update(dis_fake_loss.item(), batch_size)
        dis_real_losses.update(dis_real_loss.item(), batch_size)
        dis_losses.update(dis_loss.item(), batch_size)
        content_losses.update(content_loss.item(), batch_size)
        adversarial_losses.update(adversarial_loss.item(), batch_size)
        gen_losses.update(gen_loss.item(), batch_size)
        
        ### Logging
        showed_values = ('%18s'*1 + '%18g'*6) % \
            ('%g/%g' % (epoch, num_epochs - 1), dis_fake_losses.avg, dis_fake_losses.avg, 
            dis_losses.avg, content_losses.avg, adversarial_losses.avg, gen_losses.avg)
        pbar.set_description(showed_values)
        if i == len(pbar) - 1:
            LOGGER.info(showed_values)
        if i > 2:
            break

def val_epoch(generator: nn.Module,
              dataloader: DataLoader,
              psnr_metric: nn.Module,
              ssim_metric: nn.Module,
              batch_size: int,
              LOGGER: logging,
              ):
    psnrs = AverageMeter('PSNR', ':6.6f')
    ssims = AverageMeter('SSIM', ':6.6f')
    generator.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (hr_images, lr_images) in pbar:
            with amp.autocast():
                sr_images = generator(lr_images)
            psnr = psnr_metric(sr_images, hr_images)
            ssim = ssim_metric(sr_images, hr_images)

            ## Track values
            psnrs.update(psnr.item(), batch_size)
            ssims.update(ssim.item(), batch_size)

            ### Logging
            showed_values = ('%18s'*5 + '%18g'*2) % ('', '', '', '', '', psnrs.avg, ssims.avg)
            pbar.set_description(showed_values)
            if i == len(pbar) - 1:
                LOGGER.info(showed_values)

def test():
    return

def inference():
    return

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