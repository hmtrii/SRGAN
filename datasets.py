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
                    height:int, upscaled_factor:float, norm_minus_one_to_one:bool=False):
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
        self.norm_minus_one_to_one = norm_minus_one_to_one
        self.image_paths = get_image_paths_cyclegan_dataset(self.root_dir, self.subsets, width, height)
        self.width = width
        self.height = height
        self.upscaled_factor = upscaled_factor
        self.scaled_width = int(self.width / upscaled_factor)
        self.scaled_height = int(self.height / upscaled_factor)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index, norm_minus_one_to_one=False):
        image = Image.open(self.image_paths[index])
        hr_image = T.RandomCrop((self.width, self.height))(image)
        lr_image = hr_image.resize((self.scaled_width, self.scaled_height), resample=Image.BICUBIC)
        hr_tensor = T.ToTensor()(hr_image)
        lr_tensor = T.ToTensor()(lr_image)
        if self.norm_minus_one_to_one:
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
