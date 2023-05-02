# -- misc --
import os,math,tqdm,sys
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

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

# -- import stnls --
import stnls

class DnlsLoss(nn.Module):

    def __init__(self,ws, wt, ps, ps_dists, k, stride0, dist_crit="l1",
                 search_input="noisy", alpha = 0.5, nepochs=-1, k_decay=1.,
                 ps_dist_sched="None",ws_sched="None",epoch_ratio=1.,
                 center_crop=0.):
        super().__init__()

        # -- search info --
        self.ws = ws
        self.wt = wt
        self.ps = ps
        self.ps_dists = ps_dists
        self.k = k
        self.k0 = k
        self.stride0 = stride0
        self.nepochs = nepochs
        self.k_decay = k_decay
        self.search_input = search_input
        self.alpha = alpha
        self.dist_crit = dist_crit
        self.ps_dists_sched = ps_dist_sched
        self.ws_sched = ws_sched
        # print(self.ps_dists_sched,self.ws_sched)
        self.center_crop = center_crop
        self.curr_k = k
        self.epoch_ratio = epoch_ratio
        self.setup_ws_sched()

    def setup_ws_sched(self):
        ws = self.ws
        self.ws_grid = []
        if self.ws_sched != "None":
            if self.ws_sched.split("_")[0] == "lin":
                ws_tgt = int(self.ws_sched.split("_")[1])
                assert ws_tgt > ws
                m = (ws_tgt-ws+1)/self.nepochs
                self.ws_grid = [ws + x*m for x in np.arange(self.nepochs)]
                self.ws_grid = [int(x) for x in self.ws_grid]
                # print(self.ws_grid)

    def get_k(self,curr_epoch):
        k = self.k
        if self.k_decay > 0:# and self.search_input in ["l2","l2_v5","l2]:
            k = int(k * ((self.nepochs - curr_epoch) / self.nepochs)*self.k_decay)
            k = max(k,2)
        self.curr_k = k
        return k

    def get_ws(self,curr_epoch):
        ws = self.ws
        if len(self.ws_grid) > 0:
            ws = self.ws_grid[curr_epoch]
        self.curr_ws = ws
        return ws

    def get_ps_dists(self,curr_epoch):
        ps_dists = self.ps_dists
        if self.ps_dists_sched != "None":
            switch_epoch = int(self.ps_dists_sched.split("_")[0])
            if curr_epoch >= switch_epoch:
               ps_dists = int(self.ps_dists_sched.split("_")[1])
        self.curr_ps_dists = ps_dists
        return ps_dists

    def get_search_fxns(self,curr_epoch):
        # nsearch = 10
        # nsearch = self.ws*self.ws* (2*self.wt + 1)
        k = self.get_k(curr_epoch)
        ws = self.get_ws(curr_epoch)
        search = stnls.search.NonLocalSearch(ws,self.wt,self.ps,k,
                                             nheads=1,dist_type="l2",
                                             stride0=self.stride0,
                                             anchor_self=True)
        wr,kr = 1,1.
        # todo: try not sorting; e.g. k == -1
        ps = self.ps_dists if self.ps_dists >= 0 else self.ps
        refine = stnls.search.RefineSearch(self.ws,ps,-1,wr,kr,nheads=1,
                                           dist_type="l2",stride0=self.stride0,
                                           anchor_self=True)
        # refine = stnls.search.QuadrefSearch(self.ws,self.ps,self.k,wr,kr,nheads=1,
        #                                     dist_type="l2",stride0=self.stride0,
        #                                     anchor_self=True)
        return search,refine
    def get_search(self):
        search = stnls.search.NonLocalSearch(self.ws,self.wt,self.ps,self.curr_k,
                                             nheads=1,dist_type="l2",
                                             stride0=self.stride0,
                                             anchor_self=True)
        return search

    def get_search_video(self,noisy,deno):
        srch = None
        if self.search_input == "noisy":
            srch = noisy
        elif self.search_input == "deno":
            srch = deno
        elif self.search_input == "interp":
            srch = self.alpha * noisy + (1 - self.alpha) * deno
        else:
            raise ValueError(f"Uknown search video [{self.search_input}]")
        return srch

    def compute_loss(self,deno,noisy,dists,inds,refine,curr_epoch):
        if self.dist_crit == "l1":
            dists,_ = refine(deno,noisy,inds)
            eps = 1.*1e-6
            loss = th.mean(th.sqrt(dists+eps))
            return loss
        elif self.dist_crit == "l2":
            dists,_ = refine(deno,noisy,inds)
            if self.center_crop > 0:
                H,W = deno.shape[-2:]
                dists,inds = self.run_center_crop(dists,inds,H,W)
            loss = th.mean(dists[...,1:])
            return loss
        elif self.dist_crit == "l2_v2":
            dists,_ = refine(deno,noisy,inds)
            dists_k = th.mean(dists.detach(),-1,keepdim=True)
            dists_k = th.exp(-dists_k)
            # dists_k /= th.sum(dists_k)
            # dists_k = 10 * dists_k
            # print("dists_k.shape: ",dists_k.shape)
            loss = th.mean(dists_k * dists)
            return loss
        elif self.dist_crit == "l2_v3":
            dists,_ = refine(deno,noisy,inds)
            sigma = 30./255
            loss = th.mean((dists-sigma**2)**2)
            return loss
        elif self.dist_crit == "l2_v4":
            loss = compute_patch_k4_loss(noisy,deno,inds,self.ps)
            return loss
        elif self.dist_crit == "l2_v5":
            # print("-"*20)
            # print(dists[0,0,0])
            _,inds = remove_self(dists,inds,self.curr_k)
            inds = inds[...,:self.curr_k,:].contiguous()
            # dists,_ = refine(deno,deno,inds)
            # print(dists[0,0,0])
            dists,_ = refine(deno,noisy,inds)
            # print(dists[0,0,0])
            # exit(0)
            loss = th.mean(dists[...,1:])
            return loss
        elif self.dist_crit == "l2_v6":
            _,inds = remove_self(dists,inds,inds.shape[-2])
            _,inds = refine(deno,deno,inds) # update order
            # search = self.get_search()
            # print(inds)
            # print(inds.shape)
            # inds = inds[...,:self.curr_k,:].contiguous()
            loss = mse_with_biases(noisy,deno,inds,self.ps)
            return loss
        elif self.dist_crit == "l2_v7":
            _,inds = remove_self(dists,inds,inds.shape[-2])
            _,inds = refine(deno,deno,inds) # update order
            dists,inds = refine(deno,noisy,inds) # actual dists
            # search = self.get_search()
            # print(inds)
            # print(inds.shape)
            # inds = inds[...,:self.curr_k,:].contiguous()
            loss = mse_with_biases(noisy,deno,inds,self.ps)
            return loss
        elif self.dist_crit == "l2_v8":
            loss = mse_with_biases(noisy,deno,inds,self.ps)
            return loss
        elif self.dist_crit == "l2_v9":
            ps_dists = self.get_ps_dists(curr_epoch)
            loss = mse_with_biases(noisy,deno,inds,ps_dists)
            return loss
        elif self.dist_crit == "l2_v10":
            ps_dists = self.get_ps_dists(curr_epoch)
            loss = mse_without_biases(noisy,deno,inds,ps_dists)
            return loss
        elif self.dist_crit == "l2_v11":
            ps_dists = self.get_ps_dists(curr_epoch)
            Lambda = (curr_epoch / (1.*self.nepochs)) * self.epoch_ratio
            loss0 = mse_without_biases(noisy,deno,inds,ps_dists) # splendid-water
            # dists,_ = refine(deno,noisy,inds) # add me next.
            # loss0 = th.mean(dists[...,1:])
            loss1 = mse_with_biases(noisy,deno,inds,ps_dists)
            loss = loss0 + Lambda * loss1
            return loss
        elif self.dist_crit == "l2_v12":
            ps_dists = self.get_ps_dists(curr_epoch)
            Lambda = (curr_epoch / (1.*self.nepochs)) * self.epoch_ratio
            loss = mse_with_without_biases(noisy,deno,inds,ps_dists,Lambda)
            return loss
        elif self.dist_crit == "l2_v13":
            ps_dists = self.get_ps_dists(curr_epoch)
            loss = compute_sims_image(noisy,deno,inds,ps_dists)
            return loss
        elif self.dist_crit == "l2_v14":
            dists,_ = refine(deno,noisy,inds)
            if self.center_crop > 0:
                H,W = deno.shape[-2:]
                dists,inds = self.run_center_crop(dists,inds,H,W)
            loss = th.mean(dists)
            return loss
        elif self.dist_crit == "l2_v15":
            ps_dists = self.get_ps_dists(curr_epoch)
            Lambda = 2
            loss0 = mse_without_biases(noisy,deno,inds,ps_dists) # splendid-water
            # dists,_ = refine(deno,noisy,inds) # add me next.
            # loss0 = th.mean(dists[...,1:])
            loss1 = mse_with_biases(noisy,deno,inds,ps_dists)
            loss = loss0 + Lambda * loss1
            return loss
        else:
            raise ValueError(f"Uknown criterion [{self.dist_crit}]")

    def run_center_crop(self,dists,inds,H,W):

        # -- get size --
        nH = (H-1)//self.stride0+1
        nW = (W-1)//self.stride0+1

        # -- reshape --
        dists = rearrange(dists,'b 1 (t nH nW) k -> b t nH nW k',nH=nH,nW=nW)
        inds = rearrange(inds,'b 1 (t nH nW) k tr -> b t nH nW k tr',nH=nH,nW=nW)

        # -- center crop --
        cc = self.center_crop
        sH,eH = cc,-cc
        sW,eW = cc,-cc
        mask = th.zeros_like(dists)
        mask[:,:,sH:eH,sW:eW] = 1.
        dists_cc = dists * mask
        # inds_cc = inds[:,:,sH:eH,sW:eW]

        # -- reshape --
        dists_cc = rearrange(dists_cc,'b t nH nW k -> b 1 (t nH nW) k')
        # inds_cc = rearrange(inds_cc,'b t nH nW tr k -> b 1 (t nH nW) k tr')

        return dists_cc,inds#_cc

    def forward(self, noisy, deno, flows, curr_epoch):
        search,refine = self.get_search_fxns(curr_epoch)
        srch = self.get_search_video(noisy,deno)
        # print(srch.shape)
        dists,inds = search(srch,srch,flows.fflow,flows.bflow)

        # print(dists)
        # print("-"*50)
        # dists,_ = refine(deno,noisy,inds)
        # dists,_ = refine(srch,srch,inds)
        # print("-="*30)
        # print("-="*30)
        # print(dists.shape)
        # print(dists[0])
        # print(dists[1])
        # print(dists)
        # exit(0)
        # print("deno [max,min]: ",th.max(deno).item(),th.min(deno).item())
        # print("deno [max,min]: ",th.max(noisy).item(),th.min(noisy).item())
        # deno_d = deno.detach()
        loss = self.compute_loss(deno,noisy,dists,inds,refine,curr_epoch)
        # loss = self.compute_loss(dists[...,1:])
        return loss


def compute_patch_k4_loss(noisy,deno,inds,ps):


    unfold = stnls.UnfoldK(ps)

    patches0 = unfold(deno,inds)
    patches1 = unfold(noisy,inds)

    shape_str = 'B Q K 1 (HD C) ph pw -> K (B HD) Q (C ph pw)'
    patches0 = rearrange(patches0,shape_str,HD=1)
    patches1 = rearrange(patches1,shape_str,HD=1)

    # -- compute loss --
    delta = patches0[0] - patches1[1] + patches0[1].detach() - patches0[2].detach()
    loss = th.mean(delta**2)

    return loss


def remove_self(dists,inds,K):

    # print(inds[0])
    # print("dists.shape,inds.shape: ",dists.shape,inds.shape)
    # print(dists)

    # -- unpack patches --
    dists,inds = stnls.nn.remove_same_frame(dists,inds)
    dists = dists.contiguous()
    inds = inds.contiguous()
    # print("dists.shape,inds.shape: ",dists.shape,inds.shape)

    # -- rearrange --
    B,HD,Q,K = dists.shape
    dists = rearrange(dists,'b hd q k -> (b hd q) k')
    inds = rearrange(inds,'b hd q k tr -> (b hd q) k tr')
    # print("dists.shape,inds.shape: ",dists.shape,inds.shape)

    # -- topk --
    descending = False
    dists,inds = stnls.nn.topk_f.standard_topk(dists,inds,K,descending)
    # print(dists)

    # -- rearrange --
    dists = rearrange(dists,'(b hd q) k -> b hd q k',b=B,hd=HD)
    inds = rearrange(inds,'(b hd q) k tr -> b hd q k tr',b=B,hd=HD)

    return dists,inds

def mse_with_biases(noisy,deno,inds,ps):

    # -- unpack patches --
    # dists =th.zeros_like(inds[...,0])*1.
    # _,inds = stnls.nn.remove_same_frame(dists,inds)
    # print("inds.shape: ",inds.shape)
    inds = inds[:,0]

    unfold = stnls.UnfoldK(ps)

    patches0 = unfold(deno,inds)
    patches1 = unfold(noisy,inds)

    shape_str = 'B Q K 1 (HD C) ph pw -> K (B HD) Q (C ph pw)'
    patches0 = rearrange(patches0,shape_str,HD=1)
    patches1 = rearrange(patches1,shape_str,HD=1)

    # -- compute loss --
    delta0 = patches0[:1] - patches1[1:]
    delta1 = patches0[:1].detach() - patches0[1:].detach()
    delta = delta0 - delta1
    loss = th.mean(delta**2)
    # loss = th.mean(delta0[0]**2) + th.mean(delta**2)

    # delta = patches0[1:] - patches1[1:]
    # delta0 = patches0[:1] - patches0[1:]
    # # delta1 = patches0[:1].detach() - patches1[:1].detach()
    # # # delta = patches0[1:] - patches1[1:]
    # # delta = delta0 - delta1
    # loss = th.mean(delta0**2)

    return loss

def mse_without_biases(noisy,deno,inds,ps):

    # -- unpack patches --
    # dists =th.zeros_like(inds[...,0])*1.
    # _,inds = stnls.nn.remove_same_frame(dists,inds)
    # print("inds.shape: ",inds.shape)
    inds = inds[:,0]
    inds = inds.contiguous()

    unfold = stnls.UnfoldK(ps)

    patches0 = unfold(deno,inds)
    patches1 = unfold(noisy,inds)

    shape_str = 'B Q K 1 (HD C) ph pw -> K (B HD) Q (C ph pw)'
    patches0 = rearrange(patches0,shape_str,HD=1)
    patches1 = rearrange(patches1,shape_str,HD=1)

    # -- compute loss --
    delta0 = patches0[:1] - patches1[1:]
    # delta1 = patches0[:1].detach() - patches0[:1].detach()
    delta = delta0# + delta1
    loss = th.mean(delta**2)

    return loss

def mse_with_without_biases(noisy,deno,inds,ps,Lambda):
    inds = inds[:,0]
    inds = inds.contiguous()

    unfold = stnls.UnfoldK(ps)

    patches0 = unfold(deno,inds)
    patches1 = unfold(noisy,inds)

    shape_str = 'B Q K 1 (HD C) ph pw -> K (B HD) Q (C ph pw)'
    patches0 = rearrange(patches0,shape_str,HD=1)
    patches1 = rearrange(patches1,shape_str,HD=1)

    # -- compute loss --
    delta0 = patches0[:1] - patches1[1:]
    delta1 = patches0[:1].detach() - patches0[:1].detach()
    loss = th.mean(delta0**2) + Lambda * th.mean((delta0-delta1)**2)

    return loss



def compute_sims_image(noisy,deno,inds,ps):

    # -- view --
    # print("noisy.shape: ",noisy.shape)
    # print("deno.shape: ",deno.shape)
    # print("inds.shape: ",inds.shape)
    # print("ps: ",ps)

    # -- init --
    inds = inds[:,0]
    adj = 0
    K = inds.shape[-2]
    unfold = stnls.UnfoldK(ps,adj=adj,reflect_bounds=True)
    fold = stnls.iFoldz(noisy.shape,adj=adj)

    # -- get patches --
    loss = 0
    for k in range(K):
        patches_k = unfold(noisy,inds[...,[k],:])
        vid_k,wvid_k = fold(patches_k)
        vid_k = vid_k / wvid_k
        loss += th.mean((vid_k - deno)**2)/K

    return loss
