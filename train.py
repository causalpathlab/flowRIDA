import sys

import gcan.gcan

sys.path.append("/home/BCCRC.CA/ssubedi/projects/experiments/gcan")

import logging

import anndata as an
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import gcan

sample = "pbmc"
wdir = "znode/pbmc/"

adata = an.read_h5ad(wdir + "data/" + sample + "_pbmc1.h5ad")

gcan_object = gcan.gn.create_gcan_object(adata, wdir)

params = {
    "device": "cuda",
    "input_dim": adata.shape[1],
    "latent_dim": 15,
    "layers": [1000, 500, 100, 15],
    "learning_rate": 0.001,
    "epochs": 1000,
    "batch_size": 128,
}


gcan_object.set_nn_params(params)
gcan_object.train()
gcan_object.eval(device="cuda", eval_batch_size=100)
