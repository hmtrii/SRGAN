import os
import glob
from PIL import Image
from typing import List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T


class CustomDataset(Dataset):
    def __init__(self, width:int, height:int, upscaled_factor:int):
        super(CustomDataset, self).__init__()
        self.width = width
        self.height = height
        self.upscaled_factor = upscaled_factor
        self.scaled_width = int(self.width / upscaled_factor)
        self.scaled_height = int(self.height / upscaled_factor)
        # self.image_paths = self.get_image_paths()
    
    def get_image_paths(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> List[torch.Tensor]:
        raise NotImplementedError


class TrainSetCycleGan(CustomDataset):
    def __init__(self, root_dir:str, subsets:list, width:int, 
                    height:int, upscaled_factor:float, norm_minus_one_to_one:bool=False):
        super().__init__(root_dir, width, height, upscaled_factor)
        self.root_dir = root_dir
        self.subsets = subsets
        self.norm_minus_one_to_one = norm_minus_one_to_one
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        paths = []
        for subset in self.subsets:
            for path in glob.glob(f'{self.root_dir}/{subset}/*/*/*.jpg'):
                image = Image.open(path)
                if image.size[0] > self.width and image.size[1] > self.height and image.mode != 'L':
                    paths.append(path)
        return paths

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        hr_image = T.RandomCrop((self.width, self.height))(image)
        lr_image = hr_image.resize((self.scaled_width, self.scaled_height), resample=Image.BICUBIC)
        hr_tensor = T.ToTensor()(hr_image)
        lr_tensor = T.ToTensor()(lr_image)
        if self.norm_minus_one_to_one:
            hr_tensor = hr_tensor.mul(2.0).sub(1.0)
        return hr_tensor, lr_tensor


class TestSetCycleGan(CustomDataset):
    def __init__(self, root_dir:str, subsets:list):
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
    
    def get_image_paths(self):
        paths = []
        for subset in self.subsets:
            for path in glob.glob(f'{self.root_dir}/{subset}/*/*/*.jpg'):
                image = Image.open(path)
                if image.size[0] > self.width and image.size[1] > self.height and image.mode != 'L':
                    paths.append(path)
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return image
