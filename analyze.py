

import sys

import gcan.gcan 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/gcan')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import gcan
import torch
import logging


import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pylab as plt


from matplotlib import cm, colors

colors_10 = list(map(colors.to_hex, cm.tab10.colors))
colors_10[2] = "#279e68"  # green
colors_10[4] = "#aa40fc"  # purple
colors_10[8] = "#b5bd61"  # kakhi

# see 'category20' on https://github.com/vega/vega/wiki/Scales#scale-range-literals
colors_20 = list(map(colors.to_hex, cm.tab20.colors))

colors_20 = [
    *colors_20[0:14:2],
    *colors_20[16::2],
    *colors_20[1:15:2],
    *colors_20[17::2],
    "#ad494a",
    "#8c6d31",
]

def get_colors(N):
    if N<=10:
        return colors_10
    elif N<=20:
        return colors_20
    
def plot_umap_df(df_umap,col,fpath,pt_size=1.0,ftype='png'):
	
	nlabel = df_umap[col].nunique()
	custom_palette = get_colors(nlabel) 
 

	if ftype == 'pdf':
		fname = fpath+'_'+col+'_'+'umap.pdf'
	else:
		fname = fpath+'_'+col+'_'+'umap.png'
	
	legend_size = 7
		
	p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
		geom_point(size=pt_size) +
		scale_color_manual(values=custom_palette)  +
		guides(color=guide_legend(override_aes={'size': legend_size})))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
	
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)
 
sample = 'pbmc'
wdir = 'znode/pbmc/'

adata = an.read_h5ad(wdir+'data/'+sample+'_pbmc1.h5ad')

gcan_object = gcan.gn.create_gcan_object(adata,wdir)

params = {'device': 'cuda', 'input_dim': adata.shape[1],'latent_dim': 15, 'layers': [100,50,15], 'learning_rate': 0.001, 'epochs': 100,'batch_size':128}


gcan_object.plot_loss()


import umap
	
dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']

df_theta = pd.read_csv(wdir+'results/df_theta.txt.gz',index_col=0)
 

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_theta)
df_umap= pd.DataFrame()
df_umap['cell'] = df_theta.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_gcan_theta',pt_size=1.0,ftype='png')

