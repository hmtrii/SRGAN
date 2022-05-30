from torch import nn
from torchvision.models.vgg import vgg19  

class ContentLoss(nn.Module):
    def __init__(self) -> None:
        super(ContentLoss).__init__()
        feature_extractor = vgg19
    
    def forward(self, hr_images, lr_images):
        return