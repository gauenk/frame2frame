

# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- io --
from dev_basics import arch_io

# -- net --
from .net import DnCNN

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #        Config
    #
    # -=-=-=-=-=-=-=-=-=-=-

    # -- init --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda:0")

    # -- unpack local vars --
    local_pairs = {"io":io_pairs(),
                   "arch":arch_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)

    # -- end init --
    if econfig.is_init: return


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #     Construct Network Configs
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- init model --
    model = DnCNN(cfgs.arch.channels, cfgs.arch.num_of_layers)

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- device --
    model = model.to(device)

    return model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Helpers
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def arch_pairs():
    pairs ={"channels":3,
            "num_of_layers":17}
    return pairs

def io_pairs():
    pairs ={"pretrained_path":"",
            "pretrained_root":"",
            "pretrained_type":"",
            "pretrained_load":False,
    }
    return pairs

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)
