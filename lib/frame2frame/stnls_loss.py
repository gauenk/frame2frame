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

# -- import stnls --
import stnls
from torch.nn.functional import softmax


# -- save examples --
from dev_basics.utils import vid_io
def save_example(vid,weight,dists):
    B,HD,T,nH,nW,K = weight.shape
    B,HD,T,nH,nW,K = dists.shape
    weight = weight[:,0,:,:,:,0].view(B,T,1,nH,nW)
    dists = dists[:,0,:,:,:,0].view(B,T,1,nH,nW)
    dists = th.exp(-dists)
    mask_vid = weight*dists
    mask_vid /= mask_vid.max()
    print(mask_vid.max(),mask_vid.min(),vid.max(),vid.min(),vid.shape)
    save_vid = mask_vid * vid
    vid_io.save_video(save_vid,'output/stnls_loss','masked_dists')

def save_stack(vid,inds,stride0):
    print("inds.shape: ",inds.shape)
    stacking = stnls.agg.NonLocalStack(1,stride0)
    stack = stacking(vid,th.ones_like(inds[...,0]),inds)
    print("stack.shape: ",stack.shape)
    B,HD,K,T,F,H,W = stack.shape
    for ki in range(K):
        save_vid = th.abs(stack[:,0,ki]-vid)
        save_vid = save_vid/save_vid.max()
        vid_io.save_video(save_vid,'output/stnls_loss','stack_%d'%ki)
    exit()

def get_sr_mask(flow,stride0,H,W):
    from stnls.utils import flow2inds
    inds = flow2inds(flow,stride0)
    # print(inds[...,1] % 2)
    # inds = rearrange(inds,'(b hd) t nh nw k tr -> b hd t nh nw k tr',b=B)
    # mask0 = (inds[...,1] % 4 != 2)#+(inds[...,1] % 4 == 3)+(inds[...,1] % 4 == 1)
    # mask1 = (inds[...,2] % 4 != 2)#+(inds[...,2] % 4 == 3)+(inds[...,2] % 4 == 1)
    mask0 = (inds[...,1] % 4) == 0
    mask1 = (inds[...,2] % 4) == 0
    # mask0 = th.logical_and(((inds[...,1] % 4) >= 1.),((inds[...,1] % 4) <= 3)).float()
    # mask1 = th.logical_and(((inds[...,2] % 4) >= 1.),((inds[...,2] % 4) <= 3)).float()
    # mask0 = (inds[...,1] % 4 != 2)*(inds[...,1] % 4 != 1)
    # mask1 = (inds[...,2] % 4 != 2)*(inds[...,2] % 4 != 1)
    # mask0 = (inds[...,1] % 4 == 0)+(inds[...,1] % 4 == 3)+(inds[...,1] % 4 == 1)
    # mask1 = (inds[...,2] % 4 == 0)+(inds[...,2] % 4 == 3)+(inds[...,2] % 4 == 1)
    mask = (mask0>0).float() * (mask1>0).float()
    # mask = th.logical_and((inds[...,1] % 4 == 0),(inds[...,2] % 4 == 0))
    return mask

class WrapDnlsLoss(th.nn.Module):

    def __init__(self,loss_fxn,isize,nepochs,nbatch_sample,
                 use_flow=False,flow_method="cv2"):
        super().__init__()
        self.nepochs = nepochs
        self.nbatch_sample = nbatch_sample
        self.loss_fxn = loss_fxn
        self.use_flow = use_flow
        self.flow_method = flow_method
        self.isize = [int(x) for x in isize.split("_")]
        self.nf = 3

    def forward(self,model,optim,sched,noisy,clean):
        noisy = noisy.clone()
        from dev_basics.utils.metrics import compute_psnrs
        from data_hub.cropping import run_rand_crop

        # -- return info --
        info = edict()
        keys = ["lr","loss"]
        for key in keys: info[key] = []

        # -- train loop --
        T = noisy.shape[1]
        for epoch in range(self.nepochs):
            print("epoch: ",epoch)
            for ti in range(T-self.nf+1):
                print("ti: ",ti)

                # -- init --
                model.zero_grad()
                info.lr.append(optim.param_groups[-1]['lr'])

                # -- prepare batch --
                B = self.nbatch_sample
                noisy_i,clean_i = [],[]
                _noisy_i,_clean_i = noisy[:,ti:ti+self.nf],clean[:,ti:ti+self.nf]
                # noisy_i,clean_i = _noisy_i,_clean_i
                # noisy_i,clean_i = run_rand_crop([noisy_i,clean_i],self.isize)
                for bi in range(B):
                    noisy_i_b,clean_i_b = run_rand_crop([_noisy_i,_clean_i],self.isize)
                    noisy_i.append(noisy_i_b)
                    clean_i.append(clean_i_b)
                noisy_i = th.cat(noisy_i)
                clean_i = th.cat(clean_i)

                # -- forward --
                deno_i = model(noisy_i)
                # noisy_i = noisy_i[...,5:-5,5:-5]
                # deno_i = deno_i[...,5:-5,5:-5]
                # clean_i = clean_i[...,5:-5,5:-5]
                flows_i = flow.orun(deno_i,self.use_flow,ftype=self.flow_method)
                loss = self.loss_fxn(noisy_i, clean_i, deno_i, flows_i, epoch)

                # -- debug --
                # model = model.eval()
                # with th.no_grad():
                #     deno_i = model(noisy_i)
                # print(compute_psnrs(deno_i,clean_i,div=1.).mean())
                # # model = model.train()

                # -- logging --
                info.loss.append(loss.item())

                # -- update --
                loss.backward()
                optim.step()
                sched.step()

        return info

class DnlsLoss(nn.Module):

    def __init__(self,ws, wt, ps, ps_dists, k, stride0, dist_crit="l1",
                 search_input="deno", alpha = 0.5, nepochs=-1, k_decay=1.,
                 ps_dist_sched=None,ws_sched=None,epoch_ratio=1.,
                 dist_mask=-1,center_crop=0.,sigma=30.,nmz_bwd=False,
                 ps_scale=0.99993,ps_final=1):
        super().__init__()

        # -- search info --
        self.ws = ws
        self.wt = wt
        self.ps = ps
        self.ps_dists = ps_dists
        self.dist_mask = float(dist_mask)
        self.k = k
        self.k0 = k
        self.stride0 = stride0
        self.nepochs = nepochs
        self.k_decay = k_decay
        self.search_input = search_input
        self.alpha = alpha
        self.alpha_scale = 0.9999
        self.dist_crit = dist_crit
        self.ps_dist_sched = ps_dist_sched
        self.ws_sched = ws_sched
        self.ps_scale = ps_scale
        self.ps_final = ps_final
        # print(self.ps_dist_sched,self.ws_sched)
        self.center_crop = center_crop
        self.curr_k = k
        self.epoch_ratio = epoch_ratio
        self.sigma = sigma
        self.setup_ws_sched()
        self.name = "stnls"
        self.nmz_bwd = nmz_bwd
        # print("TODO: Center images around 0, not 0.5."+"\n"*10)

    def setup_ws_sched(self):
        ws = self.ws
        self.ws_grid = []
        if self.ws_sched != "None" and not(self.ws_sched is None): # lin_21
            if self.ws_sched.split("_")[0] == "lin":
                ws_tgt = int(self.ws_sched.split("_")[1])
                assert ws_tgt > ws
                m = (ws_tgt-ws+1)/self.nepochs
                self.ws_grid = [ws + x*m for x in np.arange(self.nepochs)]
                self.ws_grid = [int(x) for x in self.ws_grid]

    def get_k(self,curr_epoch):
        k = self.k
        if self.k_decay > 0:# and self.search_input in ["l2","l2_v5","l2]:
            k = int(k * ((self.nepochs - curr_epoch) / self.nepochs)*self.k_decay)
            k = max(k,2)
        self.curr_k = k
        return k

    def get_ps(self,step):
        ps = self.ps
        alpha = self.ps_scale**step
        ps = alpha * self.ps  + (1 - alpha) * self.ps_final
        ps = int(round(ps))
        ps = max(ps,self.ps_final)
        if (ps % 2 == 0): ps = ps+1 # make odd if even
        # print("ps [alpha,ps,step]: ",alpha,ps,step)
        # if len(self.ps_grid) > 0:
        #     ps = self.ps_grid[curr_epoch]
        return ps

    def get_ws(self,curr_epoch):
        ws = self.ws
        if len(self.ws_grid) > 0:
            ws = self.ws_grid[curr_epoch]
        self.curr_ws = ws
        return ws

    def get_ps_dists(self,curr_epoch):
        ps_dists = self.ps_dists
        if self.ps_dist_sched != "None" and not(self.ps_dist_sched is None):
            switch_epoch = int(self.ps_dist_sched.split("_")[0])
            if curr_epoch >= switch_epoch:
               ps_dists = int(self.ps_dist_sched.split("_")[1])
        self.curr_ps_dists = ps_dists
        return ps_dists

    def get_search_fxns(self,curr_epoch):
        # nsearch = 10
        # nsearch = self.ws*self.ws* (2*self.wt + 1)
        # k = self.get_k(curr_epoch)
        # ws = self.get_ws(curr_epoch)
        k = self.k
        ws = self.ws
        ps = self.get_ps(curr_epoch)
        # print(k,ws,ps)
        search = stnls.search.NonLocalSearch(ws,self.wt,ps,k,
                                             nheads=1,dist_type="l2",
                                             stride0=self.stride0,
                                             self_action="remove_ref_frame",
                                             topk_mode="each",
                                             normalize_bwd=self.nmz_bwd,
                                             full_ws=True,
                                             itype="float")
        wr,kr = 1,1.
        ps = self.ps_dists if self.ps_dists > 0 else self.ps
        k = -1
        refine = stnls.search.RefineSearch(ws,self.wt,wr,k,kr,ps,nheads=1,
                                           dist_type="l2",stride0=self.stride0,
                                           normalize_bwd=self.nmz_bwd,
                                           itype="float")
        # refine = stnls.search.QuadrefSearch(self.ws,self.ps,self.k,wr,kr,nheads=1,
        #                                     dist_type="l2",stride0=self.stride0,
        #                                     anchor_self=True)
        return search,refine
    def get_search(self):
        search = stnls.search.NonLocalSearch(self.ws,self.wt,self.ps,self.curr_k,
                                             nheads=1,dist_type="l2",
                                             self_action="remove_ref_frame",
                                             topk_mode="each")
        return search

    def get_search_video(self,noisy,deno,clean,step):
        srch = None
        if self.search_input == "noisy":
            srch = noisy
        elif "noisy-g" in self.search_input:
            sigma = int(self.search_input.split("-")[-1])
            srch = clean + th.randn_like(clean)*(sigma/255.)
        elif self.search_input == "deno":
            srch = deno
        elif self.search_input == "interp":
            alpha = self.alpha * self.alpha_scale**step
            # print("alpha: ",alpha)
            srch = alpha * noisy + (1 - alpha) * deno
        elif self.search_input == "clean":
            srch = clean
        else:
            raise ValueError(f"Uknown search video [{self.search_input}]")
        return srch

    def compute_loss(self,noisy, clean, deno, flows,curr_epoch):
        if self.dist_crit == "v0":
            assert self.dist_mask > 0.
            F = deno.shape[-3]
            search,refine = self.get_search_fxns(curr_epoch)
            srch = self.get_search_video(noisy,deno,clean,curr_epoch)
            dists0,inds = search(srch,srch,flows.fflow,flows.bflow)
            dists0 = dists0.detach()/(search.ps**2*F)
            # print(dists0.mean())
            mask = (dists0 < self.dist_mask).float()#th.exp(-10*dists0.detach())
            dists,_ = refine(deno,noisy,inds)
            # print(dists0.shape,dists.shape)
            # save_example(clean,weight,dists)
            # save_stack(clean,inds,search.stride0)
            return th.mean(mask*dists)
        elif self.dist_crit == "v0_sr":

            # -- init --
            import stnls
            assert self.dist_mask > 0.
            F = deno.shape[-3]
            search,refine = self.get_search_fxns(curr_epoch)

            # -- compute searching flows --
            wt = search.wt
            refine.ps = 1
            stride0 = search.stride0
            flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)

            # -- round flows --
            flows = flows.round().int()
            search.itype="int"
            refine.itype="int"

            # -- search --
            srch = self.get_search_video(noisy,deno,clean,curr_epoch)
            dists0,inds = search(srch,srch,flows)
            dists0 = dists0.detach()/(search.ps**2*F)
            # print(dists0.mean())
            mask = (dists0 < self.dist_mask).float()#th.exp(-10*dists0.detach())
            # print("[a] mask.sum(): ",mask.sum().item())
            mask = mask * get_sr_mask(inds.detach(),stride0,*noisy.shape[-2:])
            # print("[b] mask.sum(): ",mask.sum().item())
            # mask[...,::4,:,:] = 0
            # mask[...,:,::4,:] = 0
            # print("[c] mask.sum(): ",mask.sum().item())
            # dmask = th.zeros_like(deno)
            # dmask[...,::4,::4] = 1

            dists,_ = refine(deno,noisy,inds)
            # dists,_ = refine(deno,clean,inds)
            # print(dists0.shape,dists.shape)
            # save_example(clean,weight,dists)
            # save_stack(clean,inds,search.stride0)
            # return th.mean(mask*dists)
            return th.mean((dists+1e-6).sqrt())

            # -- viz --
            # from dev_basics.utils import vid_io
            # vid_io.save_video(clean,"output/saved_examples","clean")
            # vid_io.save_video(noisy,"output/saved_examples","noisy")
            # vid_io.save_video(deno,"output/saved_examples","deno")
            # exit()
            # return th.mean(((deno-clean)**2 + 1e-6).sqrt())
        elif self.dist_crit in ["ssims","v1"]:
            from .ssim import ssim
            F = deno.shape[-3]
            search,refine = self.get_search_fxns(curr_epoch)
            srch = self.get_search_video(noisy,deno,clean,curr_epoch)
            dists0,inds = search(srch,srch,flows.fflow,flows.bflow)
            assert search.stride0 == 1,"Must be stride0==1"
            stacking = stnls.agg.NonLocalStack(1,search.stride0)
            stack = stacking(noisy,th.ones_like(inds[...,0]),inds)
            # print("stack.shape: ",stack.shape)
            K = stack.shape[2]
            window_size = 11
            loss = 0
            deno_comp = rearrange(deno,'b t c h w -> (b t) c h w')
            for ki in range(K):
                stack_ki = rearrange(stack[:,0,ki],'b t c h w -> (b t) c h w')
                loss += th.mean((deno_comp-stack_ki)**2)
                loss += -ssim(deno_comp, stack_ki, window_size)
            return loss
        elif self.dist_crit == "global_smoothing":
            F = deno.shape[-3]
            search,refine = self.get_search_fxns(curr_epoch)
            srch = self.get_search_video(noisy,deno,clean,curr_epoch)

            dists0,inds = search(srch,srch,flows.fflow,flows.bflow)
            inds = self.global_smoothing(inds,search.wt)

            dists0,_ = refine(deno,deno,inds)
            dists0 = dists0.detach()/(search.ps**2*F)
            weight = (dists0 < self.dist_mask).float()#th.exp(-10*dists0.detach())

            dists,_ = refine(deno,noisy,inds)
            B,HD,T,nH,nW,K = dists.shape

            sH,sW = 5,5
            eH,eW = nH-5,nW-5
            dists = dists[:,:,:,sH:eH,sW:eW]
            weight = weight[:,:,:,sH:eH,sW:eW]

            return th.mean(weight*dists)/F
        else:
            raise ValueError(f"Uknown criterion [{self.dist_crit}]")

    def global_smoothing(self,inds,wt):

        # -- shapes --
        W_t = 2*wt+1
        B,HD,T,nH,nW,K,_ = inds.shape
        # inds = inds.view(B,HD,T,nH,nW,W_t-1,-1,3)

        # -- extract --
        inds0,flow = inds[...,[0]],inds[...,1:]

        # -- smoothing --
        # for ti in range(T):
        #     for wi in range(W_t-1):
        #         flow_wi = th.mean(flow[:,:,ti,:,:,wi,:,:],dim=(3,4,),keepdim=True)
        #         flow[:,:,ti,:,:,wi,:,:] = flow_wi.repeat(1,1,1,nH,nW,1,1)
        sH,sW = 5,5
        eH,eW = nH-5,nW-5
        mH,mW = eH-sH,eW-sW
        flow_c = flow[:,:,:,sH:eH,sW:eW]
        flow_c = th.mean(flow_c,dim=(3,4,),keepdim=True)
        flow[:,:,:,sH:eH,sW:eW] = flow_c.repeat(1,1,1,mH,mW,1,1)

        # -- re-create --
        inds = th.cat([inds0,flow],-1).view(B,HD,T,nH,nW,-1,3)

        return inds

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

    def forward(self, noisy, clean, deno, flows, curr_epoch):
        return self.compute_loss(noisy, clean, deno, flows,curr_epoch)
        F = deno.shape[-3]
        search,refine = self.get_search_fxns(curr_epoch)
        srch = self.get_search_video(noisy,deno,clean,curr_epoch)
        dists0,inds = search(srch,srch,flows.fflow,flows.bflow)
        dists0 = dists0.detach()/(search.ps**2*F)
        weight = (dists0 < 1e-6).float()#th.exp(-10*dists0.detach())
        dists,_ = refine(deno,noisy,inds)
        # print(dists0.shape,dists.shape)
        return th.mean(weight*dists)/F
        # loss = self.compute_loss(deno,noisy,dists,inds,refine,curr_epoch)
        # return loss


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

    inds = inds[:,0]
    unfold = stnls.UnfoldK(ps)

    patches0 = unfold(deno,inds)
    patches1 = unfold(noisy,inds)

    shape_str = 'B Q K 1 (HD C) ph pw -> K (B HD) Q (C ph pw)'
    patches0 = rearrange(patches0,shape_str,HD=1)
    patches1 = rearrange(patches1,shape_str,HD=1)

    # -- compute loss with correct grad --
    delta0 = patches0[:1] - patches1[1:]
    delta1 = patches0[:1].detach() - patches0[1:].detach()
    delta = (delta0 - delta1)**2

    # -- weights across k --
    weights = softmax(-th.mean(delta1**2,-1,keepdim=True),0)
    # weights = softmax(-10*th.mean(delta1**2,-1,keepdim=True),0)
    print(weights.shape)
    loss = th.mean(weights*delta)

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
