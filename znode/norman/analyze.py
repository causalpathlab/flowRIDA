

import sys

import flowrida
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/flowrida')

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
import umap
import scanpy as sc	


sample = 'norman'
wdir = 'znode/'+sample+'/'
tag='control'


adata = an.read_h5ad(wdir+'results/'+sample+'_'+tag+'_flowrida.h5ad')


sc.pp.neighbors(adata, use_rep="latent")
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden'])
plt.tight_layout()
plt.savefig(wdir+'/results/'+tag+'_latent_umap.png')
plt.close()

tag='pert'
adata = an.read_h5ad(wdir+'results/'+sample+'_'+tag+'_flowrida.h5ad')

sc.pp.neighbors(adata, use_rep="latent")
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['guide_merged'])
plt.tight_layout()
plt.savefig(wdir+'/results/'+tag+'_latent_umap.png')
plt.close()

sc.pp.neighbors(adata, use_rep="theta")
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['guide_merged'])
plt.tight_layout()
plt.savefig(wdir+'/results/'+tag+'_theta_umap.png')
plt.close()