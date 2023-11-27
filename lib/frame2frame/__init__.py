
# -- lightning --
from . import lightning

# -- loseses --
from .warped_loss import WarpedLoss
from .stnls_loss import DnlsLoss
from .losses import get_loss_fxn

# -- nets --
from . import dncnn
from . import fastdvdnet

#
# -- extract model --
#
from .utils import optional

def extract_model_config(cfg):
    return extract_config(cfg)

def extract_config(cfg):
    mtype = optional(cfg,'net_name','dncnn')
    if mtype in ["dncnn"]:
        return dncnn.extract_config(cfg)
    elif mtype in ["fastdvd","fastdvdnet","fdvdnet","fdvd"]:
        return fastdvdnet.extract_config(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")

def load_model(cfg):
    mtype = optional(cfg,'net_name','dncnn')
    print("load_model:" ,mtype)
    if mtype in ["dncnn"]:
        return dncnn.load_model(cfg)
    elif mtype in ["fastdvd","fastdvdnet","fdvdnet","fdvd"]:
        return fastdvdnet.load_model(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
