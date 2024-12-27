

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


import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pylab as plt


sample = 'norman'
wdir = 'znode/'+sample+'/'

# adata = an.read_h5ad(wdir+'data/'+sample+'_control.h5ad')
adata = an.read_h5ad(wdir+'data/'+sample+'_pert.h5ad')


flowrida_object = flowrida.fr.create_flowrida_object(adata,wdir)


flowrida_object.plot_loss()


import umap
	

df_theta = pd.read_csv(wdir+'results/df_theta.txt.gz',index_col=0)

df_theta = df_theta.loc[adata.obs.index.values,:]
adata.obsm['latent'] = df_theta

import scanpy as sc

sc.pp.neighbors(adata, use_rep="latent")
sc.tl.leiden(adata)
sc.tl.umap(adata)


# sc.pl.umap(adata, color=['leiden'])
sc.pl.umap(adata, color=['guide_merged'])
plt.tight_layout()
plt.savefig(wdir+'/results/umap.png')
plt.close()