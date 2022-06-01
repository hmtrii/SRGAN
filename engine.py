from typing import List
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from record import AverageMeter, ProgressMeter


def train(discriminator: nn.Module,
          generator: nn.Module,
          dataloader: DataLoader,
          optimizerD: optim.Optimizer,
          optimizerG: optim.Optimizer,
          adversarial_criterion: nn.BCELoss,
          content_criterion: nn.Module,
          adversarial_weight: float,
          batch_size: int,
          device: torch.device,
        ) -> None:
    num_batches = len(dataloader)
    epoch_time = AverageMeter("Time", ":6.6f")
    dis_fake_losses = AverageMeter('Discriminator loss for real samples', ':6.6f')
    dis_real_losses = AverageMeter('Discriminator loss for fake samples', ':6.6f')
    dis_losses = AverageMeter('Dicriminator loss', ':6.6f')
    content_losses = AverageMeter('Content loss', ':6.6f')
    adversarial_losses = AverageMeter('Adversarial loss', ':6.6f')
    gen_losses = AverageMeter('Generator loss', ':6.6f')
    process = ProgressMeter(
        num_batches, 
        [
            dis_fake_losses, dis_real_losses, dis_losses, content_losses, \
            adversarial_losses, gen_losses
        ]
    )

    discriminator.train()
    generator.train()

    start = time.time()
    batch_index = 0
    for hr_images, lr_images in dataloader:
        ### Discriminator network
        optimizerD.zero_grad()
        
        # Create lable for classification task
        real_labels = torch.full([batch_size, 1], 1.0, dtype=hr_images.dtype, device=device)
        fake_labels = torch.full([batch_size, 1], 0.0, dtype=lr_images.dtype, device=device)

        # Real samples
        prob_hr = discriminator(hr_images)
        dis_real_loss = adversarial_criterion(prob_hr, real_labels)
        dis_real_loss.backward()

        # Generate sample 
        sr_images = generator(lr_images)
        prob_sr = discriminator(sr_images.detach().clone())
        dis_fake_loss = adversarial_criterion(prob_sr, fake_labels)
        dis_fake_loss.backward()

        dis_loss = (dis_fake_loss + dis_real_loss)
        optimizerD.step()

        ### Generator network
        optimizerG.zero_grad()
        content_loss = content_criterion(hr_images, sr_images)
        adversarial_loss = adversarial_criterion(discriminator(sr_images), real_labels)
        gen_loss = content_loss + adversarial_weight*adversarial_loss
        gen_loss.backward()
        optimizerG.step()

        dis_fake_losses.update(dis_fake_loss, batch_size)
        dis_real_losses.update(dis_real_loss, batch_size)
        dis_losses.update(dis_loss, batch_size)
        content_losses.update(content_loss, batch_size)
        adversarial_losses.update(adversarial_loss, batch_size)
        gen_losses.update(gen_loss, batch_size)
        process.display(batch_index)

        batch_index += 1
    
    epoch_time.update(time.time() - start)

def val():
    return

def test():
    return