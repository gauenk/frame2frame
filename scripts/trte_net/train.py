"""

Unsupervised Training with Frame2Frame

Compare the impact of train/test using flow/nls methods

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- clearing --
import shutil
from pathlib import Path

# -- testing --
from dev_basics.trte import train

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    exps,uuids = cache_io.train_stages.run("exps/trte_net/train.cfg",
                                           ".cache_io_exps/trte_net/train/",
                                           update=True)
    print(exps[0])
    print(len(exps))
    def clear_fxn(num,cfg): return True
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_net/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_net/train.pkl",
                                records_reload=False,use_wandb=True,
                                proj_name="f2f_train_net")
    # -- view --
    print(len(results))
    if len(results) == 0: return
    print(results.columns)

if __name__ == "__main__":
    main()

