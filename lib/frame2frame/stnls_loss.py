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

    def __init__(self,ws, wt, ps, k, stride0, dist_crit="l1",
                 search_input="noisy", alpha = 0.5):
        super().__init__()
        self.search = stnls.search.NonLocalSearch(ws,wt,ps,k,nheads=1,
                                                  dist_type="l2",stride0=stride0,
                                                  anchor_self=True)
        wr,kr = 1,1.
        self.refine = stnls.search.RefineSearch(ws,ps,k,wr,kr,nheads=1,
                                                dist_type="l2",stride0=stride0,
                                                anchor_self=True)
        self.search_input = search_input
        self.alpha = alpha
        self.dist_crit = dist_crit

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

    def compute_loss(self,dists):
        if self.dist_crit == "l1":
            eps = 1.*1e-3
            loss = th.mean(th.sqrt(dists+eps))
            print(loss)
            return loss
        elif self.dist_crit == "l2":
            loss = th.mean(dists)
            return loss
        else:
            raise ValueError(f"Uknown criterion [{self.crit}]")

    def forward(self, noisy, deno, flows):
        srch = self.get_search_video(noisy,deno)
        _,inds = self.search(srch,srch,flows.fflow,flows.bflow)
        dists,_ = self.refine(deno,noisy,inds)
        loss = self.compute_loss(dists[...,1:])
        return loss

