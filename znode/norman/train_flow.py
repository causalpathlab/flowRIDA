

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
adata_p = an.read_h5ad(wdir+'results/'+sample+'_'+tag+'_flowrida.h5ad')
tag='control'
adata_c = an.read_h5ad(wdir+'results/'+sample+'_'+tag+'_flowrida.h5ad')



##### generate data from flow map

####control
adata_c = adata_c[adata_c.obs.sample(n=1000).index.values,:]

####perturbation
selected_p = 'KLF1+ctrl'
adata_p = adata_p[adata_p.obs['guide_merged']==selected_p,:]
adata_p = adata_p[adata_p.obs.sample(n=1000).index.values,:]


####

tag='flow'
flowrida_object = flowrida.fr.create_flowrida_object(adata_c,wdir,tag)

params = {
    'device': 'cuda', 
    'input_dim': adata_c.obsm['latent'].shape[1],
    'condition_dim': adata_p.obsm['latent'].shape[1],
    'hidden_dim': 128,
    'num_layers': 3,
    'learning_rate': 0.001, 
    'epochs':1000,
    'batch_size':128
    }

flowrida_object.set_flow_nn_params(params)

flowrida_object.train_flow(adata_c,adata_p)

# flowrida_object.plot_loss('cinn')

flowrida_object.flow_nn_params['device']='cpu'
flowrida_object.eval_flow(adata_c,adata_p)


import scanpy as sc 


for i, ad in enumerate([adata_c,adata_p]):
    flow = np.vstack([ad.obsm['latent'], ad.obsm['reconstructed']])
    labels = ['latent'] * ad.obsm['latent'].shape[0] + ['recons'] * ad.obsm['reconstructed'].shape[0]

    new_adata = sc.AnnData(obs={'label': labels})
    new_adata.obsm['flow'] = flow

    sc.pp.neighbors(new_adata, use_rep='flow')
    sc.tl.umap(new_adata)
    sc.pl.umap(new_adata, color='label')
    plt.tight_layout()
    plt.savefig(wdir+'/results/cinn_recons_umap_'+str(i)+'.png')
    plt.close()



