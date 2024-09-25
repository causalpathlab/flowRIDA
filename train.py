

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/gcan')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import gcan
import torch
import logging

sample = 'pbmc'
wdir = 'znode/pbmc/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_pbmc1.h5ad')

gcan_object = gcan.create_gcan_object(
	{'pbmc1':batch1,
	 },
	wdir)


params = {'device': 'cuda', 'batch_size': 128,'latent_dim': 15, 'encoder_layers': [100,15], 'learning_rate': 0.001, 'epochs': 1}

gcan_object.estimate_neighbour(params['neighbour_method'])

# def train():
gcan_object.set_nn_params(params)
gcan_object.train()
gcan_object.plot_loss()
