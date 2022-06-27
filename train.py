import os
import time
import shutil
import argparse
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.optim import lr_scheduler


from datasets import TrainSetCycleGan
from model import Generator, Discriminator
from losses import ContentLoss
from engine import train_epoch, val_epoch

from general import init_loger, create_train_dir, load_configs, random_seed, standard_time
from metrics import PSNR, SSIM


ROOT_OUTPUT = './runs'

if __name__ == '__main__':
    random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', help='Path to config file', default='./configs/srgan_cfg.yml')
    args = parser.parse_args()

    configs = load_configs(args.config_file)
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
    lr = configs['lr']
    lr_scheduler_step_size = int(num_epochs / 2) # ref paper
    lr_scheduler_gamma = configs['lr_scheduler_gamma']
    resume_discriminator = configs['resume_discriminator']
    resume_generator = configs['resume_generator']

    output_dir = create_train_dir(ROOT_OUTPUT, save_dir)
    LOGGER = init_loger(output_dir)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    shutil.copy(args.config_file, output_dir)

    train_set = TrainSetCycleGan(root_dataset, train_subsets, width_image_transform, height_image_transform, upscaled_factor)
    val_set = TrainSetCycleGan(root_dataset, val_subsets, width_image_transform, height_image_transform, upscaled_factor)
    # test_set = TrainSetCycleGan(root_dataset, test_subset)
    print('Number of train image: ', len(train_set))
    print('Number of val image: ', len(val_set))
    LOGGER.info(f'Number of train image: {len(train_set)}')
    LOGGER.info(f'Number of val image: {len(val_set)}')

    generator = Generator().to(device)
    discriminator = Discriminator(width_image_transform, height_image_transform).to(device)
    
    optimizerG = torch.optim.Adam(generator.parameters(), lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr)

    schedulerG = lr_scheduler.StepLR(optimizerG, lr_scheduler_step_size, lr_scheduler_gamma)
    schedulerD = lr_scheduler.StepLR(optimizerD, lr_scheduler_step_size, lr_scheduler_gamma)

    content_criterion = ContentLoss(ith_pool, jth_cnv).to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    # adversarial_criterion = nn.BCELoss().to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size,)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    psnr_metric = PSNR().to(device)
    ssim_metric = SSIM().to(device)

    scaler = amp.GradScaler()

    best_psnr = float('-inf')
    best_ssim = float('-inf')
    start_epoch = 0
    if resume_discriminator:
        checkpoint = torch.load(resume_discriminator, map_location=device)
        start_epoch = checkpoint['epoch']
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizer'])
        schedulerD.load_state_dict(checkpoint['scheduler'])
        best_psnr = checkpoint['bestpsnr']
        best_ssim = checkpoint['best_ssim']
    if resume_generator:
        checkpoint = torch.load(resume_generator)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer'])
        schedulerG.load_state_dict(checkpoint['scheduler'])
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']

    start = time.time()
    for epoch in range(start_epoch, num_epochs):
        show_train = ('%18s'*9) % ('Epoch', 'Prob(hr)', 'Prob(sr)', 'D(hr)', 'D(sr)', 'D_loss', 'content_loss', 'adversarial_loss', 'G_loss')
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
                    LOGGER,
                    writer,
        )

        show_val = ('%18s'*7) % ('', '', '', '', '', 'PSNR', 'SSIM')
        print(show_val)
        LOGGER.info(show_val)
        psnr, ssim = val_epoch(generator,
                               val_loader,
                               psnr_metric,
                               ssim_metric,
                               batch_size,
                               device,
                               epoch,
                               LOGGER,
                               writer,
        )
                        
        schedulerG.step()
        schedulerD.step()

        ### Save checkpoints
        torch.save({'epoch': epoch,
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                    'optimizer': optimizerG.state_dict(),
                    'scheduler': schedulerG.state_dict(),
                    'mode_state_dict': generator.state_dict()},
                    os.path.join(output_dir, 'generator_last.pth'))
        torch.save({'epoch': epoch,
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                    'optimizer': optimizerD.state_dict(),
                    'scheduler': schedulerD.state_dict(),
                    'mode_state_dict': discriminator.state_dict()},
                    os.path.join(output_dir, 'discriminator_last.pth'))
        msg = f'SAVE LAST MODELS AT EPOCH {epoch}'
        print(msg)
        LOGGER.info(msg)
        if psnr > best_psnr and ssim > best_ssim:
            best_psnr = psnr
            best_ssim = ssim
            torch.save({'epoch': epoch,
                        'psnr': psnr,
                        'ssim': ssim,
                        'optimizer': optimizerG.state_dict(),
                        'scheduler': schedulerG.state_dict(),
                        'model_state_dict': generator.state_dict()},
                        os.path.join(output_dir, 'generator_best.pth'))
            torch.save({'epoch': epoch,
                        'psnr': psnr,
                        'ssim': ssim,
                        'optimizer': optimizerD.state_dict(),
                        'scheduler': schedulerD.state_dict(),
                        'model_state_dict': discriminator.state_dict()},
                        os.path.join(output_dir, 'discriminator_best.pth'))
            best_psnr = psnr
            best_ssim = ssim
            msg = f'SAVE BEST MODELS AT EPOCH {epoch}'
            print(msg)
            LOGGER.info(msg)
    training_time = standard_time(time.time() - start)
    print(f'Complete training in {training_time}')
    LOGGER.info(f'Complete training in {training_time}')
    writer.close()
