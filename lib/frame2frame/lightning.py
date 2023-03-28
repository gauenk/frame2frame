

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

# -- losses --
from .warped_loss import WarpedLoss
from .dnls_loss import DnlsLoss

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
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

# import torch
# torch.autograd.set_detect_anomaly(True)

@econfig.set_init
def init_cfg(cfg):
    econfig.init(cfg)
    cfgs = econfig({"lit":lit_pairs(),
                    "sim":sim_pairs()})
    return cfgs

def lit_pairs():
    pairs = {"batch_size":1,"flow":True,"flow_method":"cv2",
             "isize":None,"bw":False,"lr_init":1e-3,
             "lr_final":1e-8,"weight_decay":0.,
             "nepochs":0,"task":"denoising","uuid":"",
             "scheduler":"default","step_lr_size":5,
             "step_lr_gamma":0.1,"flow_epoch":None,"flow_from_end":None,
             "ws":9,"wt":3,"ps":7,"k":5,"stride0":4,"dist_crit":"l2",
             "search_input":"interp","alpha":0.5}
    return pairs

def sim_pairs():
    pairs = {"sim_type":"g","sim_module":"stardeno",
             "sim_device":"cuda:0","load_fxn":"load_sim"}
    return pairs

def get_sim_model(self,cfg):
    if cfg.sim_type == "g":
        return None
    elif cfg.sim_type == "stardeno":
        module = importlib.load_module(cfg.sim_module)
        return module.load_noise_sim(cfg.sim_device,True).to(cfg.sim_device)
    else:
        raise ValueError(f"Unknown sim model [{sim_type}]")

class LitModel(pl.LightningModule):

    def __init__(self,lit_cfg,net,sim_model):
        super().__init__()
        lit_cfg = init_cfg(lit_cfg).lit
        for key,val in lit_cfg.items():
            setattr(self,key,val)
        self.set_flow_epoch() # only for current exps; makes last 10 epochs with optical flow.
        self.crit = self.init_crit()
        self.net = net
        self.sim_model = sim_model
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        self.automatic_optimization=True

    def forward(self,vid):
        flows = flow.orun(vid,self.flow,ftype=self.flow_method)
        deno = self.net(vid,flows=flows)
        return deno

    def sample_noisy(self,batch):
        if self.sim_model is None: return
        clean = batch['clean']
        noisy = self.sim_model.run_rgb(clean)
        batch['noisy'] = noisy

    def set_flow_epoch(self):
        if not(self.flow_epoch is None): return
        if self.flow_from_end is None: return
        if self.flow_from_end == 0: return
        self.flow_epoch = self.nepochs - self.flow_from_end

    def update_flow(self):
        if self.flow_epoch is None: return
        if self.flow_epoch <= 0: return
        if self.current_epoch >= self.flow_epoch:
            self.flow = True

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),lr=self.lr_init,
                              weight_decay=self.weight_decay)
        sched = self.configure_scheduler(optim)
        return [optim], [sched]

    def configure_scheduler(self,optim):
        if self.scheduler in ["default","exp_decay"]:
            gamma = 1-math.exp(math.log(self.lr_final/self.lr_init)/self.nepochs)
            ExponentialLR = th.optim.lr_scheduler.ExponentialLR
            scheduler = ExponentialLR(optim,gamma=gamma) # (.995)^50 ~= .78
        elif self.scheduler in ["step","steplr"]:
            args = (self.step_lr_size,self.step_lr_gamma)
            print("[Scheduler]: StepLR(%d,%2.2f)" % args)
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=self.step_lr_size,
                               gamma=self.step_lr_gamma)
        elif self.scheduler in ["cos"]:
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            T0,Tmult = 1,1
            scheduler = CosAnnLR(optim,T0,Tmult)
        elif self.scheduler in ["none"]:
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=10**3,gamma=1.)
        else:
            raise ValueError(f"Uknown scheduler [{self.scheduler}]")
        return scheduler

    def training_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- update flow --
        self.update_flow()

        # -- each sample in batch --
        loss = 0 # init @ zero
        denos,cleans = [],[]
        ntotal = len(batch['noisy'])
        nbatch = ntotal
        nbatches = (ntotal-1)//nbatch+1
        for i in range(nbatches):
            start,stop = i*nbatch,min((i+1)*nbatch,ntotal)
            deno_i,clean_i,loss_i = self.training_step_i(batch, start, stop)
            loss += loss_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatches

        # -- view params --
        # loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # -- append --
        denos = th.stack(denos)
        cleans = th.stack(cleans)

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        val_psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        self.gen_loger.info("train_psnr: %2.2f" % val_psnr)
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size)

        return loss

    def training_step_i(self, batch, start, stop):

        # -- unpack batch
        noisy = batch['noisy'][start:stop]/255.
        clean = batch['clean'][start:stop]/255.
        # print("noisy.shape: ",noisy.shape)

        # -- foward --
        deno = self.forward(noisy)

        # -- report loss --
        loss = th.mean((clean - deno)**2)
        return deno.detach(),clean,loss

    def compute_loss(clean,noisy,deno,flows):
        if self.crit_name == "warped":
            loss = self.crit.run_pairs(deno,noisy,flows)
        elif self.crit_name == "dnls":
            loss = self.crit.run_pairs(deno,noisy,flows)
        else:
            raise ValueError("Uknown loss name [{self.crit_name}]")
        return loss

    def init_crit(self):
        if self.crit_name == "warped":
            return WarpedLoss()
        else:
            return DnlsLoss(self.ws,self.wt,self.ps,self.k,self.stride0,
                            self.crit,self.search_input,self.alpha)
        else:
            raise ValueError("Uknown loss name [{self.crit_name}]")

    def validation_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        noisy,clean = batch['noisy']/255.,batch['clean']/255.

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)

        # -- report --
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)

        # -- terminal log --
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)

    def test_step(self, batch, batch_nb):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        index = float(batch['index'][0].item())
        noisy,clean = batch['noisy']/255.,batch['clean']/255.

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("index", index,on_step=True,on_epoch=False,batch_size=1)
        self.log("mem_res",  mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("mem_alloc",  mem_alloc, on_step=True, on_epoch=False, batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index#.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)



def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
