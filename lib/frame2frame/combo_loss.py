
# -- misc --
import os,math,tqdm,sys
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- import stnls --
import stnls

class ComboLoss(nn.Module):

    def __init__(self,loss0,loss1,swap=50,alpha=0.):
        super().__init__()
        self.loss0 = loss0
        self.loss1 = loss1
        self.swap = swap
        self.alpha = alpha

        # deno = self.forward(noisy)
        # loss = self.crit(self.net,noisy,deno,flows,self.current_epoch)

    def __call__(self,model,noisy,flows,epoch):
        if epoch < self.swap:
            deno,loss = self.loss0.compute(model,noisy,epoch)
        else:
            B = noisy.shape[0]
            batch = rearrange(noisy,'b t c h w -> (b t) c h w')
            deno = model(batch)
            deno = rearrange(deno,'(b t) c h w -> b t c h w',b=B)
            loss = self.loss1(noisy,deno,flows,epoch)
            if self.alpha > 1e-10:
                _,loss0 = self.loss0.compute(model,noisy,epoch)
                loss = (1 - self.alpha) * loss + self.alpha * loss0
        return deno,loss



