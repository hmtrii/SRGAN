import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms


class TrainSetCycleGan(Dataset):
    def __init__(self, root_dir:str, name_subset:list):
        super().__init__()
    
    def __len__(self):
        return

    def __getitem__(self, index):
        return 


class TestSetCycleGan(Dataset):
    def __init__(self, root_dir:str, name_subset:list):
        super().__init__()
    
    def __len__(self):
        return

    def __getitem__(self, index):
        return 