import logging
from tqdm import tqdm
from typing import List
from tensorboardX import SummaryWriter


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda import amp

from general import AverageMeter

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
          writer: SummaryWriter,
        ) -> None:
    dis_fake_losses = AverageMeter('Discriminator loss for real samples', ':6.6f')
    dis_real_losses = AverageMeter('Discriminator loss for fake samples', ':6.6f')
    dis_losses = AverageMeter('Dicriminator loss', ':6.6f')
    content_losses = AverageMeter('Content loss', ':6.6f')
    adversarial_losses = AverageMeter('Adversarial loss', ':6.6f')
    gen_losses = AverageMeter('Generator loss', ':6.6f')
    dis_hr_probs = AverageMeter('D(hr)', ':6.6f')
    dis_sr_probs = AverageMeter('D(sr)', ':6.6f')
    discriminator.train()
    generator.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (hr_images, lr_images) in pbar:
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)
        ### Discriminator network
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        optimizerD.zero_grad()
        
        # Create lable for classification task
        real_labels = torch.full([hr_images.size(0), 1], 1.0, dtype=hr_images.dtype, device=device)
        fake_labels = torch.full([hr_images.size(0), 1], 0.0, dtype=lr_images.dtype, device=device)
        
        # Real samples
        with amp.autocast():
            hr_output = discriminator(hr_images)
            dis_real_loss = adversarial_criterion(hr_output, real_labels)
        scaler.scale(dis_real_loss).backward()

        # Generate sample
        with amp.autocast():
            sr_images = generator(lr_images)
            sr_output = discriminator(sr_images.detach().clone())
            dis_fake_loss = adversarial_criterion(sr_output, fake_labels)
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

        dis_hr_prob = torch.sigmoid_(torch.mean(hr_output.detach()))
        dis_sr_prob = torch.sigmoid_(torch.mean(sr_output.detach()))

        ### Track values
        dis_fake_losses.update(dis_fake_loss.item(), hr_images.size(0))
        dis_real_losses.update(dis_real_loss.item(), hr_images.size(0))
        dis_losses.update(dis_loss.item(), hr_images.size(0))
        content_losses.update(content_loss.item(), hr_images.size(0))
        adversarial_losses.update(adversarial_loss.item(), hr_images.size(0))
        gen_losses.update(gen_loss.item(), hr_images.size(0))
        dis_hr_probs.update(dis_hr_prob.item(), hr_images.size(0))
        dis_sr_probs.update(dis_sr_prob.item(), hr_images.size(0))
        
        ### Logging
        showed_values = ('%18s'*1 + '%18g'*8) % \
            ('%g/%g' % (epoch, num_epochs - 1), dis_hr_probs.avg, dis_sr_probs.avg, dis_real_losses.avg, dis_fake_losses.avg, 
            dis_losses.avg, content_losses.avg, adversarial_losses.avg, gen_losses.avg)
        pbar.set_description(showed_values)
        if i == len(pbar) - 1:
            LOGGER.info(showed_values)
            writer.add_scalar('Train/D_fake_loss', dis_fake_losses.avg, epoch)
            writer.add_scalar('Train/D_real_loss', dis_real_losses.avg, epoch)
            writer.add_scalar('Train/D_loss', dis_losses.avg, epoch)
            writer.add_scalar('Train/content_loss', content_losses.avg, epoch)
            writer.add_scalar('Train/adversarial_loss', adversarial_losses.avg, epoch)
            writer.add_scalar('Train/G_loss', gen_losses.avg, epoch)
            writer.add_scalar('Train/D_hr_prob', dis_hr_probs.avg, epoch)
            writer.add_scalar('Train/D_sr_prob', dis_sr_probs.avg, epoch)

def val_epoch(generator: nn.Module,
              dataloader: DataLoader,
              psnr_metric: nn.Module,
              ssim_metric: nn.Module,
              batch_size: int,
              device: torch.device,
              epoch:int,
              LOGGER: logging,
              writer: SummaryWriter,
              ) -> List[float]:
    psnrs = AverageMeter('PSNR', ':6.6f')
    ssims = AverageMeter('SSIM', ':6.6f')
    generator.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (hr_images, lr_images) in pbar:
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)

            with amp.autocast():
                sr_images = generator(lr_images)
            psnr = psnr_metric(sr_images, hr_images)
            ssim = ssim_metric(sr_images, hr_images)

            ## Track values
            psnrs.update(psnr.item(), hr_images.size(0))
            ssims.update(ssim.item(), hr_images.size(0))

            ### Logging
            showed_values = ('%18s'*5 + '%18g'*2) % ('', '', '', '', '', psnrs.avg, ssims.avg)
            pbar.set_description(showed_values)
            if i == len(pbar) - 1:
                LOGGER.info(showed_values)
                writer.add_scalar('Val/psnr', psnrs.avg, epoch)
                writer.add_scalar('Val/ssim', ssims.avg, epoch)
    return psnrs.avg, ssims.avg
