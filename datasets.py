import os
import glob

from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from PIL import Image


def get_image_paths_cyclegan_dataset(root_dir, subsets):
    paths = []
    for subset in subsets:
        paths.extend(glob.glob(f'{root_dir}/{subset}/*/*/*.jpg'))
    return paths


class TrainSetCycleGan(Dataset):
    def __init__(self, root_dir:str, subsets:list, width:int, 
                    heigth:int, upscaled_factor:float):
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
        self.image_paths = get_image_paths_cyclegan_dataset(self.root_dir, self.subsets)
        self.hr_transform = T.Compose([
            T.RandomCrop((width, heigth)),
            T.ToTensor(),
        ])
        self.lr_transform = T.Compose([
            T.Resize((width//upscaled_factor, heigth//upscaled_factor), 
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
        self.image_paths = get_image_paths_cyclegan_dataset(self.root_dir, self.subsets)
        self.hr_transform = T.Compose([])
        self.lr_transform = T.Compose([])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return 

if __name__ == '__main__':
    d = TrainSetCycleGan('data/cyclegan', ['ae_photos', 'apple2orange'], 96, 96, 4)
    x = d.__getitem__(0)
    print()