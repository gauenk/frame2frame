

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
from .stnls_loss import DnlsLoss
from .nb2nb_loss import Nb2NbLoss
from .b2u_loss import B2ULoss
from .combo_loss import ComboLoss
# from .align_xform_loss import AlignXformLoss

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
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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
             "nsteps":0,"nepochs":0,"task":"denoising","uuid":"",
             "scheduler_name":"default","step_lr_size":5,
             "step_lr_gamma":0.1,"flow_epoch":None,"flow_from_end":None,
             "ws":9,"wt":3,"ps":7,"ps_dists":7,"k":5,"stride0":4,"dist_crit":"l2",
             "search_input":"deno","alpha":0.5,"crit_name":"warped","read_flows":False,
             "ntype":"g","rate":-1,"sigma":-1,"sigma_min":-1,"sigma_max":-1,
             "nb2nb_epoch_ratio":2.0,"nb2nb_lambda1":1.,"nb2nb_lambda2":1.,
             "stnls_k_decay":-1,"stnls_ps_dist_sched":"None",
             "stnls_ws_sched":"None","stnls_center_crop":0.,
             "optim_name":"adam","sgd_momentum":0.1,"sgd_dampening":0.1,
             "coswr_T0":-1,"coswr_Tmult":1,"coswr_eta_min":1e-9,
             "step_lr_multisteps":"30-50","combo_swap_epochs":50,
             "stnls_nb2nb_alpha":0.,"stnls_normalize_bwd":False,"dd_in":3,
             "dist_mask":-1,"limit_train_batches":-1,}
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
        choose_noise = data_hub.transforms.noise.choose_noise_transform
        self.noise_sim = choose_noise(lit_cfg)
        self.dset_length = 0

    def ensure_chnls(self,noisy,batch):
        if noisy.shape[-3] == self.dd_in:
            return noisy
        elif noisy.shape[-3] == 4 and self.dd_in == 3:
            return noisy[...,:3,:,:].contiguous()
        sigmas = []
        B,t,c,h,w = noisy.shape
        for b in range(B):
            sigma_b = batch['sigma'][b]/255.
            noise_b = th.ones(t,1,h,w,device=sigma_b.device) * sigma_b
            sigmas.append(noise_b)
        sigmas = th.stack(sigmas)
        return th.cat([noisy,sigmas],2)

    def forward(self,vid):
        # flows = flow.orun(vid,self.flow,ftype=self.flow_method)
        # B = vid.shape[0]
        # batch = rearrange(vid,'b t c h w -> (b t) c h w')
        deno = self.net(vid)#,flows=flows)
        # deno = rearrange(deno,'(b t) c h w -> b t c h w',b=B)
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

    # def update_flow(self):
    #     if self.flow_epoch is None: return
    #     if self.flow_epoch <= 0: return
    #     if self.current_epoch >= self.flow_epoch:
    #         self.flow = True

    def configure_optimizers(self):
        if self.optim_name == "adam":
            optim = th.optim.Adam(self.parameters(),lr=self.lr_init,
                                  weight_decay=self.weight_decay)
        elif self.optim_name == "sgd":
            optim = th.optim.SGD(self.parameters(),lr=self.lr_init,
                                 weight_decay=self.weight_decay,
                                 momentum=self.sgd_momentum,
                                 dampening=self.sgd_dampening)
        else:
            raise ValueError(f"Unknown optim [{self.optim_name}]")
        sched = self.configure_scheduler(optim)
        return [optim], [sched]

    # def lr_scheduler_step(self, scheduler, metric, idk):
    #     scheduler.step()

    def configure_scheduler(self,optim):
        if self.scheduler_name in ["default","exp_decay"]:
            gamma = math.exp(math.log(self.lr_final/self.lr_init)/self.nepochs)
            ExponentialLR = th.optim.lr_scheduler.ExponentialLR
            scheduler = ExponentialLR(optim,gamma=gamma) # (.995)^50 ~= .78
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["step","steplr"]:
            args = (self.step_lr_size,self.step_lr_gamma)
            # print("[Scheduler]: StepLR(%d,%2.2f)" % args)
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=self.step_lr_size,
                               gamma=self.step_lr_gamma)
        elif self.scheduler_name in ["cosa"] and (self.nepochs > 0):
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            scheduler = CosAnnLR(optim,self.nepochs)
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["cosa_step"] and (self.nsteps > 0):
            nsteps = self.num_steps()
            print("[CosAnnLR] nsteps: ",nsteps)
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            scheduler = CosAnnLR(optim,T_max=nsteps,eta_min=self.lr_final)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["multi_step"]:
            milestones = [int(x) for x in self.step_lr_multisteps.split("-")]
            MultiStepLR = th.optim.lr_scheduler.MultiStepLR
            scheduler = MultiStepLR(optim,milestones=milestones,
                                    gamma=self.step_lr_gamma)
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["coswr"]:
            lr_sched =th.optim.lr_scheduler
            CosineAnnealingWarmRestarts = lr_sched.CosineAnnealingWarmRestarts
            # print(self.coswr_T0,self.coswr_Tmult,self.coswr_eta_min)
            scheduler = CosineAnnealingWarmRestarts(optim,self.coswr_T0,
                                                    T_mult=self.coswr_Tmult,
                                                    eta_min=self.coswr_eta_min)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["none"]:
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=10**5,gamma=1.)
        else:
            raise ValueError(f"Uknown scheduler [{self.scheduler_name}]")
        return scheduler

    def training_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- each sample in batch --
        loss,sim = 0,0 # init @ zero
        denos,cleans = [],[]
        ntotal = len(batch['noisy'])
        nbatch = ntotal
        nbatches = (ntotal-1)//nbatch+1
        for i in range(nbatches):
            start,stop = i*nbatch,min((i+1)*nbatch,ntotal)
            deno_i,clean_i,loss_i,sim_i = self.training_step_i(batch, start, stop)
            loss += loss_i
            sim += sim_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatches
        sim = sim / nbatches

        # -- view params --
        # loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # -- append --
        denos = th.cat(denos,0)
        cleans = th.cat(cleans,0)
        # print(denos.shape,cleans.shape)
        # mse = th.mean((cleans-denos)**2).item()

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        if "fdvd" in self.crit_name:
            cleans = cleans[:,cleans.shape[1]//2]
        val_psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        # val_ssim = np.mean(compute_ssims(denos,cleans,div=1.)).item() # too slow.
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        self.log("train_psnr", val_psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        lr = self.optimizers()._optimizer.param_groups[0]['lr']
        # lr = self.optimizers().param_groups[-1]['lr']
        self.log("lr", lr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        # for i in range(1,len(sim)):
        #     self.log("sim_%d"%i, sim[i], on_step=True,
        #              on_epoch=False, batch_size=self.batch_size)
        self.log("global_step", self.global_step, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        # self.log("train_ssim", val_ssim, on_step=True,
        #          on_epoch=False, batch_size=self.batch_size)
        self.gen_loger.info("train_psnr: %2.2f" % val_psnr)

        return loss

    def training_step_i(self, batch, start, stop):

        # -- unpack batch
        noisy = batch['noisy'][start:stop]/255.
        clean = batch['clean'][start:stop]/255.
        noisy = self.ensure_chnls(noisy,batch)
        noisy = noisy[:,:,:self.dd_in]

        # -- if read flow --
        if self.read_flows:
            flows = edict({"fflow":batch['fflow'][start:stop],
                           "bflow":batch["bflow"][start:stop]})
        else:
            flows = flow.orun(noisy,self.flow,ftype=self.flow_method)

        # noisy = clean + th.randn_like(clean)*(50/255.)
        # # -- foward --
        # deno = self.forward(noisy)

        # -- non-local sim --
        sim = -1
        # if self.crit_name == "stnls":
        #     import stnls
        #     search = stnls.search.NonLocalSearch(self.ws,self.wt,self.ps,self.k,
        #                                          nheads=1,dist_type="l2",
        #                                          stride0=self.stride0,
        #                                          use_adj=False,anchor_self=True)
        #     with th.no_grad():
        #         dists = search(clean,clean,flows.fflow,flows.bflow)[0]
        #         dists = dists.reshape(-1,self.k)
        #         sim = th.quantile(dists,0.25,0).cpu().numpy()

        # -- compute fwd/loss --
        # deno = self.forward(noisy)#,flows=flows)
        # # deno = self.forward(noisy)
        # loss = th.mean((deno[:,1:] - noisy[:,:-1])**2)
        # loss = th.mean((deno[:,:-1] - noisy[:,1:])**2)
        deno,loss = self.compute_loss(clean,noisy,flows)
        return deno.detach(),clean,loss,sim

    def compute_loss(self,clean,noisy,flows):
        if self.crit_name == "warped":
            deno = self.forward(noisy)
            loss = self.crit.run_pairs(deno,noisy,flows)
        elif self.crit_name == "stnls":
            deno = self.forward(noisy)
            loss = self.crit(noisy,clean,deno,flows,self.global_step)
        elif self.crit_name == "nb2nb":
            deno,loss = self.crit.compute(self.net,noisy,self.current_epoch)
        elif self.crit_name == "b2u":
            deno,loss = self.crit.compute(self.net,noisy,self.current_epoch)
        elif self.crit_name == "nb2nb_stnls":
            deno0 = self.forward(noisy)
            loss0 = self.stnls_f2f(deno0,noisy,flows,self.current_epoch)
            deno1,loss1 = self.nb2nb.compute(self.net,noisy,self.current_epoch)
            deno = 0.5*(deno0 + deno1)
            loss = 0.5*(loss0 + loss1)
        elif self.crit_name == "sup":
            deno = self.forward(noisy)
            return deno,th.mean((deno-clean)**2)
            loss = self.crit(clean,deno)
        elif self.crit_name == "sup_fdvd":
            T = noisy.shape[1]
            deno = self.forward(noisy)
            clean = clean[:,T//2]
            return deno,th.mean((deno-clean)**2)
            loss = self.crit(clean,deno)
        elif self.crit_name == "n2n":
            deno = self.forward(noisy)
            noisy2 = self.noise_sim(clean*255)/255.
            if self.ntype in ["pg","g"]:
                noisy2 = self.noise_sim(clean)
            elif self.ntype == "msg":
                noisy2 = self.noise_sim(clean,self.noise_sim.sigma)
            else:
                raise ValueError("")
            loss = self.crit(noisy2,deno)
        elif self.crit_name == "stnls_nb2nb":
            deno,loss = self.crit(self.net,noisy,flows,self.current_epoch)
        else:
            raise ValueError("Uknown loss name [{self.crit_name}]")
        return deno,loss

    def init_crit(self):
        if self.crit_name == "warped":
            return WarpedLoss(self.dist_crit)
        elif self.crit_name == "stnls":
            return DnlsLoss(self.ws,self.wt,self.ps,self.ps_dists,self.k,self.stride0,
                            self.dist_crit,self.search_input,self.alpha,
                            self.nepochs,self.stnls_k_decay,self.stnls_ps_dist_sched,
                            self.stnls_ws_sched,1.,self.dist_mask,self.stnls_center_crop,
                            nmz_bwd=self.stnls_normalize_bwd)
        elif self.crit_name == "nb2nb":
            nepochs = self.num_epochs()
            return Nb2NbLoss(self.nb2nb_lambda1,self.nb2nb_lambda2,
                             nepochs,self.nb2nb_epoch_ratio)
        elif self.crit_name == "stnls_nb2nb":
            loss0 = Nb2NbLoss(self.nb2nb_lambda1,self.nb2nb_lambda2,
                             self.nepochs,self.nb2nb_epoch_ratio)
            loss1 = DnlsLoss(self.ws,self.wt,self.ps,self.ps_dists,self.k,self.stride0,
                             self.dist_crit,self.search_input,self.alpha,
                             self.nepochs,self.stnls_k_decay,self.stnls_ps_dist_sched,
                             self.stnls_ws_sched,1.,self.dist_mask,self.stnls_center_crop,
                             self.sigma,nmz_bwd=self.stnls_normalize_bwd)
            return ComboLoss(loss0,loss1,swap=self.combo_swap_epochs,
                             alpha=self.stnls_nb2nb_alpha)
        elif self.crit_name == "b2u":
            ninfo = "%s_%d_%d" % (self.ntype,self.sigma,self.rate)
            return B2ULoss(self.nb2nb_lambda1,self.nb2nb_lambda2,
                           self.nepochs,self.nb2nb_epoch_ratio,ninfo)
        elif self.crit_name == "nb2nb_stnls":
            nb2nb = Nb2NbLoss(self.nb2nb_lambda1,self.nb2nb_lambda2,
                              self.nepochs,self.nb2nb_epoch_ratio)
            stnls_f2f = DnlsLoss(self.ws,self.wt,self.ps,self.ps_dists,
                                 self.k,self.stride0,self.dist_crit,
                                 self.search_input,self.alpha,self.nepochs,
                                 self.stnls_k_decay,self.stnls_ps_dist_sched,
                                 self.stnls_ws_sched,1.,self.stnls_center_crop,
                                 self.sigma,nmz_bwd=self.stnls_normalize_bwd)
            self.nb2nb = nb2nb
            self.stnls_f2f = stnls_f2f
            return None
        elif self.crit_name in ["sup","n2n","sup_fdvd"]:
            def sup(clean,deno):
                if self.dist_crit == "l1":
                    return th.mean(th.abs(clean - deno))
                elif "l2" in self.dist_crit:
                    return th.mean((clean - deno)**2)
                else:
                    raise ValueError(f"Uknown dist_crit [{dist_crit}]")
            return sup
        else:
            raise ValueError(f"Uknown loss name [{self.crit_name}]")

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        if self.nsteps > 0:
            return self.nsteps
        elif self.limit_train_batches > 0:
            dataset_size = self.limit_train_batches
            num_devices = 1
        else:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            num_devices = max(1, self.trainer.num_devices)
        acc = self.trainer.accumulate_grad_batches
        num_steps = dataset_size * self.trainer.max_epochs // (acc * num_devices)
        return num_steps

    def validation_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        noisy,clean = batch['noisy']/255.,batch['clean']/255.
        val_index = batch['index'].cpu().item()
        noisy = self.ensure_chnls(noisy,batch)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        val_ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- report --
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_psnr", val_psnr, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_ssim", val_ssim, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_index", val_index, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("global_step",self.global_step,on_step=False,
                 on_epoch=True,batch_size=1)
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)
        self.gen_loger.info("val_ssim: %.3f" % val_ssim)


    def test_step(self, batch, batch_nb):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        index = float(batch['index'][0].item())
        noisy,clean = batch['noisy']/255.,batch['clean']/255.
        noisy = self.ensure_chnls(noisy,batch)

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
        self.log("test_psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_index", index,on_step=True,on_epoch=False,batch_size=1)
        self.log("test_mem_res", mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_mem_alloc", mem_alloc,on_step=True,on_epoch=False,batch_size=1)
        self.log("global_step",self.global_step,on_step=True,on_epoch=False,batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)
        self.gen_loger.info("te_ssim: %.3f" % ssim)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index#.cpu().numpy().item()
        return results

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        if self.nsteps > 0:
            return self.nsteps
        elif self.limit_train_batches > 0:
            dataset_size = self.limit_train_batches
            num_devices = 1
        else:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            num_devices = max(1, self.trainer.num_devices)
        acc = self.trainer.accumulate_grad_batches
        num_steps = dataset_size * self.trainer.max_epochs // (acc * num_devices)
        return num_steps

    def num_epochs(self) -> int:
        """Get number of epochs"""
        if self.nepochs > 0:
            return self.nepochs
        elif self.limit_train_batches > 0:
            dataset_size = self.limit_train_batches
            num_devices = 1
        else:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            num_devices = max(1, self.trainer.num_devices)
        steps_per_epoch = dataset_size/num_devices
        num_epochs = self.nsteps / steps_per_epoch
        # num_steps = dataset_size * self.trainer.max_epochs // (acc * num_devices)
        return num_epochs


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
