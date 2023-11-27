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
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class WrapWarpedLoss(th.nn.Module):

    def __init__(self,loss_fxn,isize,nepochs,nbatch_sample,
                 use_flow=False,flow_method="cv2"):
        super().__init__()
        self.nepochs = nepochs
        self.nbatch_sample = nbatch_sample
        self.loss_fxn = loss_fxn
        self.use_flow = use_flow
        self.flow_method = flow_method
        self.isize = [int(x) for x in isize.split("_")]
        print("self.use_flow: ",self.use_flow)

    def forward(self,model,optim,sched,noisy,clean):
        from data_hub.cropping import run_rand_crop

        # -- return info --
        info = edict()
        keys = ["lr","loss"]
        for key in keys: info[key] = []

        # -- train loop --
        T = noisy.shape[1]
        for epoch in range(self.nepochs):
            for ti in range(T-5):

                # -- init --
                model.zero_grad()
                info.lr.append(optim.param_groups[-1]['lr'])

                # -- prepare batch --
                B = self.nbatch_sample
                noisy_i,clean_i = [],[]
                _noisy_i,_clean_i = noisy[:,ti:ti+5],clean[:,ti:ti+5]
                for bi in range(B):
                    inputs_b = [_noisy_i,_clean_i]
                    outs = run_rand_crop([_noisy_i,_clean_i],self.isize)
                    noisy_i_b,clean_i_b,flows_i_b = outs
                    noisy_i.append(noisy_i_b)
                    clean_i.append(clean_i_b)
                noisy_i = th.cat(noisy_i)
                clean_i = th.cat(clean_i)

                # -- update --
                deno_i = model(noisy_i)
                flows_i = flow.orun(deno_i,self.use_flow,ftype=self.flow_method)
                loss = self.loss_fxn.run_pairs(deno_i, noisy_i, flows_i, epoch)
                info.loss.append(loss.item())
                loss.backward()
                optim.step()
                sched.step()

        return info

class WarpedLoss(nn.Module):
    def __init__(self,dist_crit="l2",use_stnls=False,search=None,
                 loss_type="warp",ws=9,ps=7,dist_mask=2e-1,ps_scale=1.,ps_final=1):
        super().__init__()
        # self.criterion = nn.L1Loss(size_average=False)
        # self.criterion = nn.L1Loss()
        self.dist_crit = dist_crit
        self.use_stnls = use_stnls
        self.search = search
        self.loss_type = loss_type
        self.ws = ws
        self.ps = ps
        self.ps_scale = ps_scale
        self.ps_final = ps_final
        self.dist_mask = dist_mask

    def get_ps(self,step):
        ps = self.ps
        alpha = self.ps_scale**step
        ps = alpha * self.ps  + (1 - alpha) * self.ps_final
        ps = int(round(ps))
        if (ps % 2 == 0): ps = ps+1 # make odd if even
        ps = max(ps,self.ps_final)
        # print("ps [alpha,ps,step]: ",alpha,ps,step)
        # if len(self.ps_grid) > 0:
        #     ps = self.ps_grid[curr_epoch]
        return ps

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
        # mask = th.abs(a + b) > 1001.75
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

    def forward(self, input, target, flow, step, in_mask=None):

        # flow = self.apply_stnls_correction(input,target,flow)
        # noisy, deno, flow (deno -> noisy)
        # Warp input on target
        if self.loss_type == "warp":
            warped, mask = self.warp(target, flow)
            # Compute the occlusion mask
            mask = self.occlusion_mask(warped, flow, mask)
            if not(in_mask is None):
                mask = in_mask * mask
            # Compute the masked loss
            # loss = self.criterion(mask*input, mask*warped)
            # loss = th.mean((mask*input - mask*warped)**2)
            loss = self.compute_loss((mask*input - mask*warped)**2)
        else:
            import stnls
            ws,wr,ps = self.ws,1,self.get_ps(step)
            flow = rearrange(flow,'b two h w ->  b h w 1 two')
            stride0 = input.shape[-2]//flow.shape[1]
            flow = flow.flip(-1)
            search = stnls.search.PairedRefine(ws,wr,-1,1.,ps,stride0=stride0)
            dists,_ = search(input,target,flow)
            loss = (in_mask * dists[...,0]).mean()
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

    def update_stnls_flow(self,src,tgt,flow):
        from einops import rearrange
        if not self.use_stnls:
            return th.zeros_like(flow[:,[0]]),flow
        F = src.shape[-3]
        ps = self.search.ps
        stride0 = self.search.stride0
        dists,inds = self.search(src.detach(),tgt.detach(),flow)
        inds = rearrange(inds,'b 1 nh nw 1 two ->  b two nh nw')
        inds = inds.flip(-3)
        dists = rearrange(dists,'b 1 nh nw 1 ->  b 1 nh nw')
        dists /= (ps**2*F)
        # print(th.cat([flow[0,:,:3,:3],inds[0,:,:3,:3]],-1))
        if (stride0 > 1) and (self.loss_type == "warp"):
            from torch.nn import functional as F
            dists = F.interpolate(dists,scale_factor=stride0)
            inds = F.interpolate(inds,scale_factor=stride0)
            # dists[::stride0] = 100. # invalidate according to stride0
            assert dists.shape[-2:] == src.shape[-2:]
        return dists,inds

    def run_pairs(self,deno,noisy,flows,step):

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
        wt = 1
        W_t = 2*wt+1
        import stnls
        from stnls.search.utils import get_time_window_inds
        flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,1)
        for ti in range(T):
            tgrid = get_time_window_inds(ti,wt,T)
            for _tj in range(1,W_t):
                tj = tgrid[_tj]
                if ti > tj: flow = flows[:,ti,_tj-1]
                elif ti < tj: flow = flows[:,ti,_tj-1]
                else: raise ValueError("ti != tj")
                # if ti > tj: flow = flows.bflow[:,tj]
                # elif ti < tj: flow = flows.fflow[:,tj]
                dists,flow = self.update_stnls_flow(deno[:,ti],deno[:,tj],flow)
                mask = (dists < self.dist_mask).float()
                loss += self.forward(deno[:,ti],noisy[:,tj],flow,step,mask)
        loss /= (T*(W_t-1))

        # for t in range(1,T):
        #     flow = flows.bflow[:,t]
        #     dists,flow = self.update_stnls_flow(deno[:,t],deno[:,t-1],flow)
        #     mask = (dists < 2e-1).float()
        #     loss += self.forward(deno[:,t],noisy[:,t-1],flow,step,mask)
        # for t in range(T-1):
        #     flow = flows.fflow[:,t]
        #     dists,flow = self.update_stnls_flow(deno[:,t],deno[:,t+1],flow)
        #     mask = (dists < 2e-1).float()
        #     loss += self.forward(deno[:,t],noisy[:,t+1],flow,step,mask)
        # loss /= 2*(T-1)

        return loss



