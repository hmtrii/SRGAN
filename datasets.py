import os
import glob

from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from PIL import Image


class TrainSetCycleGan(Dataset):
    def __init__(self, root_dir:str, subsets:list, width:int, 
                    heigth:int, resized_factor:float):
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
        self.image_paths = self.get_image_paths()
        self.hr_transform = T.Compose([
            T.RandomCrop((width,heigth)),
            T.ToTensor(),
        ])
        self.lr_transform = T.Compose([
            T.Resize((width//resized_factor, heigth//resized_factor), 
            interpolation=T.functional.InterpolationMode.BICUBIC),  
        ])

    def get_image_paths(self):
        paths = []
        for subset in self.subsets:
            paths.extend(glob.glob(f'{self.root_dir}/{subset}/*/*/*.jpg'))
        return paths
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        hr_tensor = self.hr_transform(image)
        lr_tensor = self.lr_transform(hr_tensor)
        return hr_tensor, lr_tensor


class TestSetCycleGan(Dataset):
    def __init__(self, root_dir:str, name_subset:list):
        super().__init__()
    
    def __len__(self):
        return

    def __getitem__(self, index):
        return 

# if __name__ == '__main__':
#     d = TrainSetCycleGan('data/cyclegan', ['ae_photos', 'apple2orange'], 96, 96, 4)
#     x = d.__getitem__(0)
#     print()