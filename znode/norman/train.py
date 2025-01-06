

import sys

sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/flowrida')
import flowrida

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import flowrida
import torch
import logging

sample = 'norman'
wdir = 'znode/'+sample+'/'

tag='pert'
adata = an.read_h5ad(wdir+'data/'+sample+'_'+tag+'.h5ad')

flowrida_object = flowrida.fr.create_flowrida_object(adata,wdir,tag)

params = {'device': 'cuda', 'input_dim': adata.shape[1],'latent_dim': 15, 'layers': [200,100,50,15], 'learning_rate': 0.001, 'epochs':500,'batch_size':128}


flowrida_object.set_nn_params(params)
flowrida_object.train()
flowrida_object.plot_loss()
flowrida_object.eval(device='cuda',eval_batch_size=1000)

adata.write_h5ad(wdir+'results/'+sample+'_'+tag+'_flowrida.h5ad')