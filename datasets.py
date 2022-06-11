import os
import glob

from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from PIL import Image


def get_image_paths_cyclegan_dataset(root_dir, subsets, min_width, min_height):
    paths = []
    for subset in subsets:
        for path in glob.glob(f'{root_dir}/{subset}/*/*/*.jpg'):
            image = Image.open(path)
            if image.size[0] > min_width and image.size[1] > min_height and image.mode != 'L':
                paths.append(path)
    return paths


class TrainSetCycleGan(Dataset):
    def __init__(self, root_dir:str, subsets:list, width:int, 
                    height:int, upscaled_factor:float):
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
        self.image_paths = get_image_paths_cyclegan_dataset(self.root_dir, self.subsets, width, height)
        self.hr_transform = T.Compose([
            T.RandomCrop((width, height)),
            T.ToTensor(),
        ])
        self.lr_transform = T.Compose([
            T.Resize((width//upscaled_factor, height//upscaled_factor), 
            interpolation=T.functional.InterpolationMode.BICUBIC),  
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index, norm_minus_one_to_one=True):
        image = Image.open(self.image_paths[index])
        hr_tensor = self.hr_transform(image)
        lr_tensor = self.lr_transform(hr_tensor)
        if norm_minus_one_to_one:
            hr_tensor = hr_tensor.mul(2.0).sub(1.0)
        return hr_tensor, lr_tensor


class TestSetCycleGan(Dataset):
    def __init__(self, root_dir:str, subsets:list):
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
        self.image_paths = get_image_paths_cyclegan_dataset(self.root_dir, self.subsets, 0, 0)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return image
