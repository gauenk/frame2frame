# -- misc --
import os,math,tqdm,sys
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat
from torch.autograd import Variable

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- caching results --
import cache_io

# # -- network --
# import nlnet

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

# -- misc --
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils.timer import ExpTimer
import dev_basics.utils.gpu_mem as gpu_mem

# -- scikit-image --
from scipy.ndimage.morphology import binary_dilation

# -- noise sims --
import importlib
# try:
#     import stardeno
# except:
#     pass

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

class WarpedLoss(nn.Module):
    def __init__(self,dist_crit="l2"):
        super().__init__()
        # self.criterion = nn.L1Loss(size_average=False)
        # self.criterion = nn.L1Loss()
        self.dist_crit = dist_crit

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow.
        Code heavily inspired by PWC-Net
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        grid = grid.cuda()
        vgrid = Variable(grid) + flo.cuda()

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/max(W-1, 1)-1.0
        vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/max(H-1, 1)-1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        # Define a first mask containing pixels that wasn't properly interpolated
        mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask.cuda(), vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output, mask

    # Computes the occlusion map based on the optical flow
    def occlusion_mask(self, warped, of, old_mask):
        """
        Computes an occlusion mask based on the optical flow
        warped: [B, C, H, W] warped frame (only used for size)
        of: [B, 2, H, W] flow
        old_mask: [B, C, H, W] first estimate of the mask
        """
        a = th.zeros_like(warped)
        b = th.zeros_like(warped)

        # Compute an occlusion based on the divergence of the optical flow
        a[:, :, :-1, :] = (of[0, 0, 1:, :] - of[0, 0, :-1, :])
        b[:, :, :, :-1] = (of[0, 1, :, 1:] - of[0, 1, :, :-1])
        mask = th.abs(a + b) > 0.75
        mask = mask.cpu().numpy()

        # Dilates slightly the occlusion map to be more conservative
        ball = np.zeros((3, 3))
        ball[1, 0] = 1
        ball[0, 1] = 1
        ball[1, 1] = 1
        ball[2, 1] = 1
        ball[1, 2] = 1
        mask[0, 0, :, :] = binary_dilation(mask[0, 0, :, :], ball)

        # Remove the boundaries (values extrapolated on the boundaries)
        mask[:, :, 0, :] = 1
        mask[:, :, mask.shape[2]-1, :] = 1
        mask[:, :, :, 0] = 1
        mask[:, :, :, mask.shape[3]-1] = 1

        # Invert the mask because we want a mask of good pixels
        mask = Variable((old_mask * torch.Tensor(1-mask).cuda()))
        return mask

    def forward(self, input, target, flow):
        # noisy, deno, flow (deno -> noisy)
        # Warp input on target
        warped, mask = self.warp(target, flow)
        # Compute the occlusion mask
        mask = self.occlusion_mask(warped, flow, mask)
        # Compute the masked loss
        # loss = self.criterion(mask*input, mask*warped)
        # loss = th.mean((mask*input - mask*warped)**2)
        loss = self.compute_loss((mask*input - mask*warped)**2)
        return loss

    def compute_loss(self,dists):
        if self.dist_crit == "l1":
            eps = 1.*1e-6
            loss = th.mean(th.sqrt(dists+eps))
            return loss
        elif self.dist_crit == "l2":
            loss = th.mean(dists)
            return loss
        else:
            raise ValueError(f"Uknown criterion [{self.crit}]")

    def run_pairs(self,deno,noisy,flows):

        # -- unpack --
        # deno = rearrange(deno,'b t c h w -> (b t) c h w')
        # noisy = rearrange(noisy,'b t c h w -> (b t) c h w')
        # fflow = rearrange(flows.fflow,'b t c h w -> (b t) c h w')
        # bflow = rearrange(flows.bflow,'b t c h w -> (b t) c h w')

        # Computes an occlusion mask based on the optical flow
        # warped: [B, C, H, W] warped frame (only used for size)
        # of: [B, 2, H, W] flow
        # old_mask: [B, C, H, W] first estimate of the mask
        T = deno.shape[1]
        loss = 0
        for t in range(1,T):
            loss += self.forward(noisy[:,t-1],deno[:,t],flows.bflow[:,t])
        for t in range(T-1):
            loss += self.forward(noisy[:,t+1],deno[:,t],flows.fflow[:,t])
        loss /= 2*(T-1)
        return loss



