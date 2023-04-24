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
                 search_input="noisy", alpha = 0.5, nepochs=-1, k_decay=1.):
        super().__init__()

        # -- search info --
        self.ws = ws
        self.wt = wt
        self.ps = ps
        self.ps_dists = ps_dists
        self.k = k
        self.stride0 = stride0
        self.nepochs = nepochs
        self.k_decay = k_decay
        self.search_input = search_input
        self.alpha = alpha
        self.dist_crit = dist_crit
        self.curr_k = k

    def get_search_fxns(self,curr_epoch):
        k = self.k
        self.curr_k = k
        if self.k_decay > 0:# and self.search_input in ["l2","l2_v5","l2]:
            k = int(k * ((self.nepochs - curr_epoch) / self.nepochs)*self.k_decay)
            k = max(k,2)
        # nsearch = 10
        # nsearch = self.ws*self.ws* (2*self.wt + 1)
        nsearch = k
        search = stnls.search.NonLocalSearch(self.ws,self.wt,self.ps,nsearch,
                                             nheads=1,dist_type="l2",
                                             stride0=self.stride0,
                                             anchor_self=True)
        wr,kr = 1,1.
        refine = stnls.search.RefineSearch(self.ws,self.ps,k,wr,kr,nheads=1,
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

    def compute_loss(self,deno,noisy,dists,inds,refine):
        if self.dist_crit == "l1":
            dists,_ = refine(deno,noisy,inds)
            eps = 1.*1e-6
            loss = th.mean(th.sqrt(dists+eps))
            return loss
        elif self.dist_crit == "l2":
            dists,_ = refine(deno,noisy,inds)
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
        elif self.dist_crit == "l2_v9":
            loss = mse_with_biases(noisy,deno,inds,self.ps_dists)
            return loss
        else:
            raise ValueError(f"Uknown criterion [{self.crit}]")

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
        loss = self.compute_loss(deno,noisy,dists,inds,refine)
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
    delta1 = patches0[:1].detach() - patches0[:1].detach()
    delta = delta0 + delta1
    loss = th.mean(delta**2)

    return loss

