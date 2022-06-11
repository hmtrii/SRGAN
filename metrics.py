import cv2
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class PSNR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, raw_images: torch.Tensor, dst_images: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((raw_images * 255.0 - dst_images * 255.0) ** 2 + 1e-8, dim=[1, 2, 3])
        psnr = 10 * torch.log10_(255.0**2 / mse)
        return torch.mean(psnr, axis=0)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, gaussian_sigma=1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_window = self.create_kernel_window()
    
    def create_kernel_window(self):
        gaussian_kernel = cv2.getGaussianKernel(self.window_size, self.gaussian_sigma)
        gaussian_kernel_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())
        return gaussian_kernel_window

    def ssim_torch(self, 
                    raw_tensor: torch.Tensor,
                    dst_tensor: torch.Tensor,
                   ) -> float:
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2

        gaussian_kernel_window = torch.from_numpy(self.gaussian_kernel_window).view(1, 1, self.window_size, self.window_size)
        gaussian_kernel_window = gaussian_kernel_window.expand(raw_tensor.size(1), 1, self.window_size, self.window_size)
        gaussian_kernel_window = gaussian_kernel_window.to(device=raw_tensor.device, dtype=raw_tensor.dtype)

        raw_mean = F.conv2d(raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=raw_tensor.shape[1])
        dst_mean = F.conv2d(dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=dst_tensor.shape[1])
        raw_mean_square = raw_mean ** 2
        dst_mean_square = dst_mean ** 2
        raw_dst_mean = raw_mean * dst_mean
        raw_variance = F.conv2d(raw_tensor * raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0),
                                groups=raw_tensor.shape[1]) - raw_mean_square
        dst_variance = F.conv2d(dst_tensor * dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0),
                                groups=raw_tensor.shape[1]) - dst_mean_square
        raw_dst_covariance = F.conv2d(raw_tensor * dst_tensor, gaussian_kernel_window, stride=1, padding=(0, 0),
                                    groups=raw_tensor.shape[1]) - raw_dst_mean
        ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
        ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (raw_variance + dst_variance + c2)
        ssim_metrics = ssim_molecular / ssim_denominator
        ssim_metrics = torch.mean(ssim_metrics, [1, 2, 3])
        return ssim_metrics

    def forward(self, raw_images: torch.Tensor, dst_images: torch.Tensor):
        raw_images = raw_images.to(torch.float64)
        dst_images = dst_images.to(torch.float64)
        return torch.mean(self.ssim_torch(raw_images, dst_images), axis=0)