
# -- python imports --
import os,math
import numpy as np
import torch as th
from einops import rearrange
from dev_basics.utils.misc import set_seed,optional
from easydict import EasyDict as edict
from dev_basics.utils import vid_io
from pathlib import Path

# -- networks --
import importlib

# -- data --
import data_hub

# -- chunking --
from dev_basics import net_chunks

# -- recording --
import cache_io

# -- losses --
from frame2frame import get_loss_fxn

# -- metrics --
import dev_basics.utils.raw as raw_utils
from dev_basics.utils.misc import optional
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred


def get_videos(cfg):

    # -- load from files --
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,-1)
    assert len(indices) == 1,"Must only be one video subsequence."
    _noisy = data[cfg.dset][indices[0]]['noisy'][None,:].to(device)/255.
    _clean = data[cfg.dset][indices[0]]['clean'][None,:].to(device)/255.

    # -- save --
    # fft_n = th.fft.fftshift(th.fft.rfft2(_noisy[:,:10].mean(-3,keepdim=True)))
    # fft_c = th.fft.fftshift(th.fft.rfft2(_clean[:,:10].mean(-3,keepdim=True)))
    # # fft_n_abs = fft_n.abs()
    # # fft_n_abs = fft_n.angle()
    # # print(fft_n.shape)
    # vid_io.save_video(20*th.log10(fft_n[:,:10].abs()),"output/saved_examples","noisy_abs")
    # vid_io.save_video(fft_n[:,:10].angle(),"output/saved_examples","noisy_angle")
    # vid_io.save_video(20*th.log10(fft_c[:,:10].abs()),"output/saved_examples","clean_abs")
    # vid_io.save_video(fft_c[:,:10].angle(),"output/saved_examples","clean_angle")
    # vid_io.save_video(_noisy[:,:10],"output/saved_examples","noisy")
    # vid_io.save_video(_clean[:,:10],"output/saved_examples","clean")
    # exit()

    # -- add noise channel --
    if optional(cfg,"dd_in",3) == 4:
        _noisy = append_sigma(_noisy,cfg.sigma)

    # -- dev subsampling --
    print("_noisy.shape: ",_noisy.shape)
    _noisy = _noisy[:,:20]
    _clean = _clean[:,:20]
    # _noisy = _noisy[:,:20,:,128:128+256,128:128+256]
    # _clean = _clean[:,:20,:,128:128+256,128:128+256]
    # _noisy = _noisy[:,:20,:,256:256,256:256]
    # _clean = _clean[:,:20,:,256:256,256:256]
    # print("noise: ",th.var(_noisy-_clean).sqrt()*255.,cfg.sigma)

    # -- info --
    print(_noisy.shape)

    # -- split --
    noisy,clean = split_vids(_noisy,_clean,cfg.num_tr_frames)

    return noisy,clean

def anscombe_fwd(vid):
    return vid
    # vid_xform = 2*th.sqrt(vid+3/8.)
    # vid_xform = vid_xform / (2*math.sqrt(1+3/8.))
    # return vid_xform

def anscombe_bwd(vid):
    return vid
    # vid_xform = vid * (2*math.sqrt(1+3/8.))
    # vid_xform = (vid_xform/2.)**2 - 3/8.
    # return vid_xform

def get_videos(cfg):
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/calibrate/")
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/curtains/")
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/floor/")
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/curtains_v3/")
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/hand_soap/")
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/shelf_bottles_v2/")
    # files = Path("/home/gauenk/Documents/data/canon2023/raw/cans_light0/")
    files = Path("/home/gauenk/Documents/data/canon2023/raw/cans_light1/")

    vid = []
    info = {}
    files = sorted(list(files.iterdir()))
    for i,fn_i in enumerate(files):

        # if i > 5: break

        # -- read --
        print(fn_i)
        raw_i,info_i = raw_utils.read_raw(str(fn_i))
        # sH,sW = 1024,int(1.5*1024)
        sH,sW = 512,int(0.5*1024)
        eH,eW = sH+2*1024,sW+2*1024
        # raw_i = raw_i[...,sH:eH,sW:eW]

        # -- raw only --
        # raw_i = raw_i[...,None,:,:]

        # -- to rgb --
        # raw_i = raw_utils.raw2rgb(raw_i)
        # raw_i = rearrange(raw_i,'h w c -> c h w')

        # -- packed --
        # print("raw_i [min,max]: ",raw_i.min().item(),raw_i.max().item())
        raw_i = raw_utils.packing(raw_i,'raw2rgb')
        # raw_i = raw_i[...,::2,::2]
        # print("[post] raw_i [min,max]: ",raw_i.min().item(),raw_i.max().item())
        raw_i[1] = (raw_i[1] + raw_i[-1])/2.
        raw_i = raw_i[:3]
        # print("raw_i.shape: ",raw_i.shape)

        # -- append --
        vid.append(th.from_numpy(raw_i))
        for k,v in info_i.items():
            if k in info: info[k].append(v)
            else: info[k] = [v]

    # -- stack --
    vid = th.stack(vid)[None,:].cuda().float()
    # vid /= vid.max()
    # print("input [min,max]: ",vid.min().item(),vid.max().item())
    vid = anscombe_fwd(vid)
    print(vid.shape)
    if vid.shape[-3] == 1:
        vid = vid.repeat(1,1,3,1,1)
    # print("vid.shape: ",vid.shape)
    print("input [min,max]: ",vid.min().item(),vid.max().item())

    # -- info --
    # vid_rgb = raw_utils.video_raw2rgb(vid)
    # for c in range(vid_rgb.shape[-3]): print(vid_rgb[:,:,c].mean().item())
    # vid_io.save_video(vid_rgb,"output/instances_adapt/","noisy")
    # exit()

    # -- split --
    noisy,clean = split_vids(vid,vid,cfg.num_tr_frames)

    return noisy,clean,info

def save_video(raw,raw_info):
    pass

def append_sigma(noisy,sigma):
    if noisy.shape[-3] == 4: return noisy
    sigma_map = th.ones_like(noisy[:,:,:1])*(sigma/255.)
    noisy = th.cat([noisy,sigma_map],2)
    return noisy

def split_vids(_noisy,_clean,num_tr):
    noisy,clean = edict(),edict()
    noisy.tr = _noisy[:,:num_tr].contiguous()
    noisy.te = _noisy[:,num_tr:].contiguous()
    clean.tr = _clean[:,:num_tr].contiguous()
    clean.te = _clean[:,num_tr:].contiguous()
    return noisy,clean

def load_model(cfg):
    device = "cuda:0"
    net_module = importlib.import_module(cfg.net_module)
    net = net_module.load_model(cfg).to(device)
    net = net.eval()
    return net

def get_scheduler(cfg,name,optim):
    lr_sched = th.optim.lr_scheduler
    if name in [None,"none"]:
        return lr_sched.LambdaLR(optim,lambda x: x)
    elif name in ["cosa"]:
        nsteps = cfg.seq_nepochs*cfg.num_tr_frames
        scheduler = lr_sched.CosineAnnealingLR(optim,T_max=nsteps)
        return scheduler
    else:
        raise ValueError(f"Uknown scheduler [{name}]")

def run_training(cfg,model,noisy,clean,raw_info):

    # -- get loss --
    if cfg.loss_type != "none":

        # -- train model (but not for BN) --
        model = model.train()
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                m.eval()
        model.apply(set_bn_eval)

        # -- optimizer --
        optim = th.optim.Adam(model.parameters(),lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
        scheduler = get_scheduler(cfg,optional(cfg,"scheduler_name",None),optim)
        # assert noisy.shape[1] == cfg.num_tr_frames,"Must be equal."
        cfg.num_tr_frames = min(noisy.shape[1],cfg.num_tr_frames)

        # -- init --
        loss_fxn = get_loss_fxn(cfg,cfg.loss_type)

        # -- run --
        train_info = loss_fxn(model,optim,scheduler,noisy,clean)
    else:
        # -- skip --
        train_info = {}

    # -- test on training data --
    model = model.eval()
    test_info = run_testing(cfg,model,noisy,clean,raw_info)
    # print(test_info)
    # exit()

    # -- info --
    info = edict()
    for key in train_info:
        info[key] = train_info[key]
    for key in test_info:
        info["tr_%s"%key] = test_info[key]

    return info

def run_testing(cfg,model,noisy,clean,raw_info):

    # -- denoised output --
    model = model.eval()
    chunk_cfg = net_chunks.extract_chunks_config(cfg)
    fwd_fxn0 = lambda vid,flows=None: model(vid)
    # fwd_fxn0 = lambda vid,flows=None: vid
    fwd_fxn = net_chunks.chunk(chunk_cfg,fwd_fxn0)
    print("noisy[min,max]: ",noisy.min().item(),noisy.max().item())
    # deno = noisy
    with th.no_grad():
        deno = fwd_fxn(noisy)
    print("th.mean((deno-noisy)**2).item(): ",th.mean((deno-noisy)**2).item())
    # print("deno.shape: ",deno.shape,clean.shape)
    print("deno [min,max]: ",deno.min().item(),deno.max().item())
    print("noisy [min,max]: ",noisy.min().item(),noisy.max().item())

    # -- save --
    print(deno.min(),deno.max(),noisy.min(),noisy.max())
    deno = anscombe_bwd(deno)
    noisy = anscombe_bwd(noisy)
    deno = deno.clip(0,1)
    deno = th.cat([deno,deno[:,:,[-1]]],2)
    deno = raw_utils.vid_packing(deno,'rgb2raw')
    print("deno.shape: ",deno.shape)
    noisy = th.cat([noisy,noisy[:,:,[-1]]],2)
    noisy = raw_utils.vid_packing(noisy,'rgb2raw')
    print("noisy.shape: ",noisy.shape)

    print("deno [min,max]: ",deno.min().item(),deno.max().item())
    print("noisy [min,max]: ",noisy.min().item(),noisy.max().item())
    deno = deno.mean(-3,keepdim=True)
    deno_rgb = raw_utils.video_raw2rgb(deno)
    # deno_rgb[...,1,:,:] = deno_rgb[...,1,:,:] * (0.483/0.154)
    # deno_rgb[...,2,:,:] = deno_rgb[...,2,:,:] * (0.337/0.559)
    for c in range(deno_rgb.shape[-3]): print(deno_rgb[:,:,c].mean().item())
    # deno_rgb = deno
    vid_io.save_video(deno_rgb,"output/instances_adapt_light1/","deno")
    noisy = noisy.mean(-3,keepdim=True)
    noisy_rgb = raw_utils.video_raw2rgb(noisy)
    for c in range(noisy_rgb.shape[-3]): print(noisy_rgb[:,:,c].mean().item())
    # noisy_rgb = noisy
    vid_io.save_video(noisy_rgb,"output/instances_adapt_light1/","noisy")
    T = noisy.shape[1]
    print("noisy.shape: ",noisy.shape)
    for ti in range(T-1):
        ave = noisy[:,ti:].mean(1,keepdim=True)
        ave_rgb = raw_utils.video_raw2rgb(ave)
        vid_io.save_video(ave_rgb,"output/instances_adapt_light1/","ave_%d"%ti)
    exit()

    # -- metrics --
    # psnrs = compute_psnrs(deno,clean,div=1.)
    # psnrs_noisy = compute_psnrs(noisy[...,:3,:,:],clean,div=1.)
    # ssims = compute_ssims(deno,clean,div=1.)
    # ssims_noisy = compute_ssims(noisy[...,:3,:,:],clean,div=1.)
    # strred = compute_strred(deno,clean,div=1.)
    # print(psnrs,psnrs_noisy)

    # -- info --
    info_te = edict()
    info_te.psnrs = np.mean(psnrs).item()
    info_te.psnrs_noisy = np.mean(psnrs_noisy).item()
    info_te.ssims = np.mean(ssims).item()
    info_te.ssims_noisy = np.mean(ssims_noisy).item()
    info_te.strred = np.mean(strred).item()
    return info_te

def run(cfg):

    # -- init --
    set_seed(cfg.seed)
    set_pretrained_path(cfg)

    # -- read data --
    noisy,clean,info = get_videos(cfg)

    # -- read model --
    model = load_model(cfg)

    # -- run testing --
    info_tr = run_training(cfg,model,noisy.tr,clean.tr,info)
    info_te = run_testing(cfg,model,noisy.te,clean.te,info)



    exit()

    # -- create results --
    results = edict()
    for k,v in info_tr.items():
        results[k] = v
    for k,v in info_te.items():
        assert not(k in results)
        results[k] = v

    return results

def set_pretrained_path(cfg):
    cfg.pretrained_path = get_pretrained_path(cfg)
    print(cfg.pretrained_path)
    name = cfg.net_name
    if name == "dncnn":
        cfg.pretrained_root = "./output/train/trte_dncnn/checkpoints"
    elif name == "fdvd":
        cfg.pretrained_root = "./output/train/trte_net/checkpoints"
    else:
        raise ValueError(f"Unknown net name [{name}]")

def get_pretrained_path(cfg):
    sigma = cfg.pretrained_sigma
    name = cfg.net_name
    if name == "dncnn":
        if sigma == 5:
            return "5debab32-d268-4b4e-b503-407963eaf767-save-global_step=1600.ckpt"
        elif sigma == 10:
            return ""
        elif sigma == 25.:
            return "78bb93e5-56ee-4261-bd83-a9a79b10284d-save-global_step=5000.ckpt"
            # return "78bb93e5-56ee-4261-bd83-a9a79b10284d-epoch=111.ckpt"
        elif sigma == 30.:
            return "b1733f3c-d8c9-4288-8698-974c7c3ab2f0-save-global_step=5000.ckpt"
        elif sigma == 35.:
            return "42e2b111-e4d7-4b31-8079-9772179f220f-save-global_step=5000.ckpt"
        elif sigma == 50:
            return ""
        else: raise ValueError(f"Unknown sigma [{sigma}]")
    elif name == "fdvd":
        if sigma == 10:
            return ""
        elif sigma == 25.:
            return "91e91c3a-6067-4291-8253-029e21a6bc3a-save-global_step=14000.ckpt"
        elif sigma == 30.:
            return "0eacdb59-d884-4bf9-b03f-787b0daed2dc-save-global_step=14000.ckpt"
        elif sigma == 35.:
            return "1b87b443-f5b1-4ceb-8937-fc3dbd84cef3-save-global_step=8000.ckpt"
        elif sigma == 50:
            return ""
        else: raise ValueError(f"Unknown sigma [{sigma}]")
    else:
        raise ValueError(f"Unknown network [{name}]")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Separate Config Grids
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def f2f_grid():
    exps = {
        "group0":{"loss_type":["f2f",],"warp_loss_type":[["warp","stnls",]]},
        "group2":{"ws":[9],"ps":[11],"stride0":[1],"ps_scale":[.99],"ps_final":[3],}
    }
    return exps

def f2f_plus_grid():
    exps = {
        "group0":{"loss_type":["f2f_plus",],"warp_loss_type":[["warp","stnls"]]},
        "group1":{"ws":[9],"ps":[11],"stride0":[1],"ps_scale":[0.99],"ps_final":[3]},
    }
    return exps

def stnls_grid():
    exps = {
        "group0":{
            "loss_type":["stnls"],
            "search_input":["deno"],
        },
        "group1":{"ws":[21],"ps":[7],"stride0":[1],"ps_scale":[0.99],"ps_final":[7],
                  "dist_mask":[2e-0],"dist_crit":["v0"]},
    }
    return exps

def none_grid():
    exps = {"group0":{"loss_type":["none"]}}
    return exps

def sup_grid():
    exps = {"group0":{"loss_type":["sup"]}}
    return exps

def collect_grids(base,learn):
    # grids = [f2f_grid,f2f_plus_grid,stnls_grid,none_grid]
    # grids = [none_grid,stnls_grid,f2f_grid]
    # grids = [none_grid,stnls_grid]
    # grids = [none_grid,stnls_grid]
    grids = [stnls_grid]
    # grids = [none_grid,]#stnls_grid]
    cfgs = []
    for grid in grids:
        exps = base | grid()
        if grid != none_grid: exps = exps | learn
        # exps = exps | learn
        cfgs += cache_io.exps.load_edata(exps)
    return cfgs

def sr_grids():
    group = {"pretrained_sigma":[],"sigma":[],'ntype':[],'sr_scale':[]}
    sigmas = [[25,-1]]
    for sigma_pre,sigma in sigmas:
        group["pretrained_sigma"].append(sigma_pre)
        group["sigma"].append(sigma)
        group['ntype'].append("sr")
        group['sr_scale'].append(4)
    return group

def sigma_grids():
    group = {"pretrained_sigma":[],"sigma":[],'ntype':[]}
    # sigmas = [[25,30],[25,35],[30,25],[30,30],[30,35],[35,30],[35,25]]
    # sigmas = [[25,25],[25,30],[25,35],[30,35],[30,30],[30,25]]#,[30,35],[35,30],[35,25]]
    # sigmas = [[25,25],[25,30],[25,35],[30,35],[30,30],[30,25]]#,[30,35],[35,30],[35,25]]
    # sigmas = [[25,25],[25,35]]#,[30,35],[35,30],[35,25]]
    sigmas = [[25,25],[30,30],[35,35]]#,[30,35],[35,30],[35,25]]
    # sigmas = [[25,25],[25,30],[25,35],[30,30],[35,35]]
    # sigmas = [[25,25],[25,30],[30,30]]
    sigmas = [[25,25]]
    # sigmas = [[30,30]]
    # sigmas = [[25,25]]
    for sigma_pre,sigma in sigmas:
        group["pretrained_sigma"].append(sigma_pre)
        group["sigma"].append(sigma)
        group['ntype'].append("g")
    return group


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Launching
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main():

    # -- init --
    print("PID: ",os.getpid())

    # -- config --
    base = {
        # v0.1
        "group10":{"tag":["v0.0001"],"seed":[123]},
        "group11":{"vid_name":[[
            # "sunflower",
            # "snowboard",
            # "tractor",
            # "hypersmooth",
            # "motorbike",
            # "park_joy",
            "rafting",
            # "touchdown"
        ]
        ],"dname":["set8"],"dset":["te"]},
        "group12":{"net_module":["frame2frame"],
                   "net_name":["dncnn"],
                   "dd_in":[3],
                   # "net_name":["fdvd"],
                   # "dd_in":[4],
                   },
        "group13":{"num_tr_frames":[9],"batch_nframes":[10],"use_flow":[True],
                   "pretrained_load":[True],"pretrained_type":["lit"]},
    }
    base_learn = {
        "group14": {"lr":[1.001e-4],"weight_decay":[1e-8],
                    "seq_nepochs":[[500]],"scheduler_name":["cosa"],
                    "spatial_chunk_size":[256],"spatial_chunk_overlap":[0.2],
                    "temporal_chunk_size":[5],"unsup_isize":["96_96"],
                    "nbatch_sample":[1]}
    }
    base['listed100'] = sigma_grids()
    # base['listed100'] = sr_grids()
    exps = collect_grids(base,base_learn)

    # -- run --
    results = cache_io.run_exps(exps,run,proj_name="instances_adapt",
                                name=".cache_io/instances_adapt",
                                records_fn=".cache_records/instances_adapt.pkl",
                                records_reload=True,
                                enable_dispatch="slurm",use_wandb=True)
    if len(results) == 0:
        print("No results")
        return

    # -- view --
    results = results.fillna(value="None")
    results = results.rename(columns={"seq_nepochs":"ne",#"warp_loss_type":"wlt",
                                      "pretrained_sigma":"p_sigma",
                                      "search_input":"si"})
    # print(results[['vid_name','loss_type','ne','wlt','p_sigma','sigma',
    #                'tr_psnrs','tr_ssims','tr_strred']])
    # print(results[['vid_name','loss_type','ne','wlt','p_sigma','sigma',
    #                'psnrs','ssims','strred']])
    # keys = ['wlt','p_sigma','sigma',
    #         'tr_psnrs','tr_ssims','tr_strred',
    #         'psnrs','ssims','strred']
    # print(results[['p_sigma','sigma','psnrs','ssims','strred']])
    # print(results[['vid_name','p_sigma','psnrs','ssims','strred']])

    keys = [
        # 'wlt',
        'p_sigma','sigma','si',
        'psnrs','ssims','strred',
        'tr_psnrs','tr_ssims',#'tr_strred',
    ]
    for group0,gdf0 in results.groupby("loss_type"):
        for group1,gdf1 in gdf0.groupby("ne"):
            print(group0,group1)
            print(gdf1[keys])


if __name__ == "__main__":
    main()
