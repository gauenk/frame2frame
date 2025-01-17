import torch
import torch.nn as nn
from einops import rearrange


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.times = {}

    def forward(self, x, flows=None):
        # -- vid to batch --
        T = -1
        if x.ndim == 5:
            B,T,F,H,W = x.shape
            x = x.reshape(B*T,F,H,W)
            # x = rearrange(x,'b t c h w -> (b t) c h w')

        out = x-self.dncnn(x)

        # -- batch to vid --
        if T != -1:
            out = out.reshape(B,T,F,H,W)
            # out = rearrange(out,'(b t) c h w -> b t c h w',t=T)
            # x = rearrange(x,'(b t) c h w -> b t c h w',t=T)

        return out
