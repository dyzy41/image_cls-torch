import timm
import torch.nn as nn

class vit224(nn.Module):
    def __init__(self, in_channels=3, num_class=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, features_only=False, in_chans=in_channels, num_classes=num_class)

    def forward(self, x):
        out = self.model(x)
        return out
