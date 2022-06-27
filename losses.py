import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vgg import vgg19
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms as T


class ContentLoss(nn.Module):
    def __init__(self, ith_pool, jth_cnv) -> None:
        super(ContentLoss, self).__init__()
        self.ith_pool = ith_pool
        self.jth_conv = jth_cnv
        self.id_layer = self.create_id_layer()
        self.feature_extractor = self.create_feature_extractor_()

    def create_id_layer(self):
        if self.ith_pool <= 2:
            id = 5 * self.ith_pool - self.jth_conv * 2
        else:
            id = (10 + 9 * (self.ith_pool - 2)) - self.jth_conv * 2
        return f'features.{id}'
    
    def create_feature_extractor_(self):
        model = vgg19(pretrained=True)
        feature_extractor = create_feature_extractor(model, [self.id_layer])
        feature_extractor.eval()
        for params in feature_extractor.parameters():
            params.requires_grad = False
        return feature_extractor

    def forward(self, hr_tensor: torch.tensor, sr_tensor: torch.tensor):
        feature_hr = self.feature_extractor(hr_tensor)[self.id_layer]
        feature_lr = self.feature_extractor(sr_tensor)[self.id_layer]
        return F.mse_loss(feature_hr, feature_lr)
