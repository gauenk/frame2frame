
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dev basics --
# from dev_basics.report import deno_report
from functools import partial
from dev_basics.aug_test import test_x8
from dev_basics import flow
from dev_basics import net_chunks
from dev_basics.utils.misc import get_region_gt
from dev_basics.utils.misc import optional,slice_flows,set_seed
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred
from dev_basics.utils import vid_io

# -- special fwd pass --
from .b2u_loss import B2ULoss

# -- config --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

def test_pairs():
    pairs = {"device":"cuda:0","seed":123,
             "frame_start":0,"frame_end":-1,"dset":"val",
             "aug_test":False,"longest_space_chunk":False,
             "flow":False,"burn_in":False,"arch_name":None,
             "saved_dir":"./output/saved_examples/","uuid":"uuid_def",
             "flow_sigma":-1,"internal_adapt_nsteps":0,
             "internal_adapt_nepochs":0,"nframes":0,"read_flows":False,
             "save_deno":True,"python_module":"dev_basics.trte.id_model",
             "bench_bwd":False,"append_noise_map":False,"arch_name":"default",
             "crit_name":"warp"}
    return pairs

def run_ub2_test(model,noisy,flows):
    b2u = B2ULoss(-1,-1,-1,-1,"")
    deno = b2u.test(model,noisy)
    return deno

@econfig.set_init
def run(cfg):

    # -- config --
    econfig.init(cfg)
    epairs = econfig.extract_pairs
    tcfg = epairs(cfg,test_pairs())
    module = econfig.required_module(tcfg,'python_module')
    model_cfg = epairs(module.extract_model_config(tcfg),cfg)
    if econfig.is_init: return
    if tcfg.frame_end == -1: tcfg.frame_end = tcfg.frame_start + cfg.nframes - 1

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- set seed --
    set_seed(tcfg.seed)

    # -- set device --
    th.cuda.set_device(int(tcfg.device.split(":")[1]))

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.strred = []
    results.noisy_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []

    # -- init keyword fields --
    time_fields = ["flow","deno","attn","extract","search",
                   "agg","fold","fwd_grad","bwd"]
    for field in time_fields:
        results["timer_%s"%field] = []
    mem_fields = ["deno","adapt","fwd_grad","bwd"]
    for field in mem_fields:
        results["%s_mem_res"%field] = []
        results["%s_mem_alloc"%field] = []


    # -- burn_in once --
    burn_in = tcfg.burn_in

    # -- load model --
    model = module.load_model(model_cfg)

    # -- data --
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[tcfg.dset],cfg.vid_name,
                                     tcfg.frame_start,tcfg.frame_end)
    print(indices)

    for index in indices:

        # -- create timer --
        timer = ExpTimer()
        memer = GpuMemer()

        # -- clean memory --
        th.cuda.empty_cache()
        # print("index: ",index)

        # -- unpack --
        sample = data[cfg.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'][None,],sample['clean'][None,]
        noisy,clean = noisy.to(tcfg.device),clean.to(tcfg.device)
        vid_frames = sample['fnums'].numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- resample noise for flow --
        if tcfg.flow_sigma >= 0:
            noisy_f = th.normal(clean,tcfg.flow_sigma)
        else:
            noisy_f = noisy

        # -- optical flow --
        with TimeIt(timer,"flow"):
            if tcfg.read_flows:
                flows = {'fflow':sample['fflow'],'bflow':sample['bflow']}
                flows = edict({k:flows[k][None,:].to(tcfg.device) for k in flows})
            else:
                flows = flow.orun(noisy_f,tcfg.flow,ftype="svnlb")
        print([flows[k].shape for k in flows])

        # -- augmented testing --
        if tcfg.aug_test:
            aug_fxn = partial(test_x8,model)#,use_refine=cfg.aug_refine_inds)
        else:
            aug_fxn = model.forward

        # -- chunked processing --
        chunk_cfg = net_chunks.extract_chunks_config(cfg)
        if tcfg.longest_space_chunk:
            set_longest_spatial_chunk(chunk_cfg,noisy.shape)
        fwd_fxn = net_chunks.chunk(chunk_cfg,aug_fxn)
        chunk_fwd = fwd_fxn

        # -- run once for setup gpu --
        if burn_in:
            with th.no_grad():
                noisy_a = noisy[[0],...,:128,:128].contiguous()
                flows_a = flow.orun(noisy_a,False)
                fwd_fxn(noisy_a/imax,flows_a)
            if hasattr(model,'reset_times'):
                model.reset_times()
        burn_in = False # only run first iteration.

        # -- internal adaptation --
        adapt_psnrs = [0.]
        run_adapt = tcfg.internal_adapt_nsteps > 0
        run_adapt = run_adapt and (tcfg.internal_adapt_nepochs > 0)
        with MemIt(memer,"adapt"):
            with TimeIt(timer,"adapt"):
                if run_adapt:
                    noisy_a = noisy[0,:5]
                    clean_a = clean[0,:5]
                    flows_a = flow.slice_at(flows,slice(0,5),1)
                    region_gt = get_region_gt(noisy_a.shape)
                    adapt_psnrs = model.run_internal_adapt(
                        noisy_a,cfg.sigma,flows=flows_a,
                        clean_gt = clean_a,region_gt = region_gt,
                        chunk_fwd=chunk_fwd)
                    if hasattr(model,'reset_times'):
                        model.reset_times()

        # -- append noise map --
        noisy_input = noisy
        if tcfg.append_noise_map:
            B,T,C,H,W = noisy.shape
            noise_map = th.ones((B,T,1,H,W),device=noisy.device)*cfg.sigma
            noisy_input = th.cat([noisy,noise_map],2)

        # -- denoise! --
        with MemIt(memer,"deno"):
            with TimeIt(timer,"deno"):
                with th.no_grad():
                    if tcfg.crit_name == "b2u":
                        deno = run_ub2_test(fwd_fxn,noisy_input/imax,flows)
                    else:
                        deno = fwd_fxn(noisy_input/imax,flows)
                deno = deno.clamp(0.,1.)*imax
        mtimes = model.times

        # -- unpack if exists --
        if hasattr(model,'mem_res'):
            if model.mem_res != -1:
                memer["deno"] = (model.mem_res,model.mem_alloc)

        # -- save example --
        out_dir = Path(tcfg.saved_dir) / tcfg.arch_name / str(tcfg.uuid)
        if tcfg.save_deno:
            print("Saving Denoised Output [%s]" % out_dir)
            deno_fns = vid_io.save_video(deno,out_dir,"deno")
        else:
            deno_fns = ["" for _ in range(deno.shape[0])]

        # -- deno quality metrics --
        noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
        psnrs = compute_psnrs(clean,deno,div=imax)
        ssims = compute_ssims(clean,deno,div=imax)
        strred = compute_strred(clean,deno,div=imax)
        # print(np.mean(psnrs),np.mean(ssims),np.mean(strred))

        # -- compare [delete me] --
        # warps = model.warps
        # print(warps.shape)
        # ref = warps[:,:,:3]
        # div = ref.max().item()
        # K = 8
        # cmps = rearrange(warps[:,:,3:],'b t (k c) h w -> k b t c h w',k=K)
        # for k in range(K):
        #     # mse = th.mean((ref - cmps[k])**2)
        #     warp_psnrs = compute_psnrs(ref,cmps[k],div=div)
        #     print(k,warp_psnrs)
        # # exit(0)
        # print(psnrs,np.mean(psnrs))
        # import vrt
        # print(deno.shape,clean.shape)
        # psnrs = vrt.calculate_psnr(deno,clean)
        # print(psnrs,np.mean(psnrs),np.mean(ssims),np.mean(strred))

        # -- measure bwd info --
        if tcfg.bench_bwd:
            measure_bwd(model,fwd_fxn,flows,noisy/imax,
                        clean/imax,timer,memer)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.strred.append(strred)
        results.noisy_psnrs.append(noisy_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        for name,(mem_res,mem_alloc) in memer.items():
            key = "%s_%s" % (name,"mem_res")
            results[key].append([mem_res])
            key = "%s_%s" % (name,"mem_alloc")
            results[key].append([mem_alloc])
        for name,time in timer.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)
        for name,time in mtimes.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    return results

def measure_bwd(model,fwd_fxn,flows,noisy,clean,timer,memer):

    # -- train mode --
    model.train()

    # -- forward pass again --
    with MemIt(memer,"fwd_grad"):
        with TimeIt(timer,"fwd_grad"):
            deno = fwd_fxn(noisy,flows)
    if hasattr(model,'reset_times'):
        model.reset_times()

    # -- backward pass! --
    with MemIt(memer,"bwd"):
        with TimeIt(timer,"bwd"):
            loss = th.mean((deno - clean)**2)
            loss.backward()

    # -- test mode again --
    model.eval()
