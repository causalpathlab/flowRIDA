import os
import numpy as np
import scanpy as sc
import anndata as ad


adata = sc.read_h5ad('norman_raw.h5ad')



needed_obs = adata.obs[["guide_identity","read_count", "UMI_count","gemgroup","good_coverage","number_of_cells","guide_ids"]].copy()


adata_new = sc.AnnData(adata.X.copy(), obs=needed_obs, var=adata.var.copy())


adata_new = adata_new[adata_new.obs["guide_identity"] != "NegCtrl1_NegCtrl0__NegCtrl1_NegCtrl0"] 


adata_new.obs["guide_merged"] = adata_new.obs["guide_identity"]


import re 
for i in np.unique(adata_new.obs["guide_merged"]):
   m = re.match(r"NegCtrl(.*)_NegCtrl(.*)+NegCtrl(.*)_NegCtrl(.*)", i)
   if m :
        adata_new.obs["guide_merged"].replace(i,"ctrl",inplace=True)
        
        
old_pool = []
for i in np.unique(adata_new.obs["guide_merged"]):
    if i == "ctrl":
        old_pool.append(i)
        continue
    split = i.split("__")[1]
    split = split.split("_")
    for j, string in enumerate(split):
        if "NegCtrl" in split[j]:
            split[j] = "ctrl"
    if len(split) == 1:
        if split[0] in old_pool:
            print("old:",i, "new:",split[0])
        adata_new.obs["guide_merged"].replace(i,split[0],inplace=True)
        old_pool.append(split[0])
    else:
        if f"{split[0]}+{split[1]}" in old_pool:
            print("old:",i, "new:",f"{split[0]}+{split[1]}")
        adata_new.obs["guide_merged"].replace(i, f"{split[0]}+{split[1]}",inplace=True)
        old_pool.append(f"{split[0]}+{split[1]}")
        
        
adata_new.obs["guide_merged"]

adata_new.obs["guide_merged"].value_counts()


adata_control = adata_new[adata_new.obs["guide_merged"]=='ctrl'].copy()


sc.pp.normalize_total(adata_control)
sc.pp.log1p(adata_control)
sc.pp.highly_variable_genes(adata_control,n_top_genes=5000, subset=True)
adata_control.write_h5ad('norman_control.h5ad',compression='gzip')



sel_pert = ['CEBPE+RUNX1T1', 'KLF1+ctrl', 'TBX3+TBX2', 'SLC4A1+ctrl', 'ETS2+CNN1', 'UBASH3B+OSR2', 'DUSP9+ETS2', 'ctrl+BAK1', 'ctrl+KLF1']


adata_pert = adata_new[adata_new.obs["guide_merged"].isin(sel_pert)].copy()


sc.pp.normalize_total(adata_pert)
sc.pp.log1p(adata_pert)
sc.pp.highly_variable_genes(adata_pert,n_top_genes=5000, subset=True)

adata_pert.write_h5ad('norman_pert.h5ad',compression='gzip')
