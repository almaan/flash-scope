#!/usr/bin/env python
# coding: utf-8

# In[48]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[49]:


import anndata as ad
import torch as t
import numpy as np
import scanpy as sc
from typing import List
import pandas as pd


# In[50]:


sc_pth = '../../cci-explore/validation/pipeline/data/common/HumanSpinalCord/sc/GSM6774327_10X162_1_W08.h5ad'
sp_pth = '../../cci-explore/validation/pipeline/data/common/HumanSpinalCord/sp/V10F24-107_C1_W8.h5ad'


# In[51]:


ad_sc = ad.read_h5ad(sc_pth)
ad_sp = ad.read_h5ad(sp_pth)


# In[52]:


label_col = 'Subtypes'


# In[53]:


def preprocess(ad_sc: ad.AnnData,
               ad_sp: ad.AnnData,
               label_col: str,
               gene_list: List[str] | None = None, 
               min_label_member: int = 20,
               min_sc_counts: int = 100,
               min_sp_counts: int = 100,
               ):

    ad_sc.var_names_make_unique()
    ad_sp.var_names_make_unique()

    vc = ad_sc.obs[label_col].value_counts()
    keep = vc.index[vc >= min_label_member].values
    ad_sc = ad_sc[np.isin(ad_sc.obs[label_col].values,keep)].copy()



    sc.pp.filter_genes(ad_sc, min_counts=min_sc_counts)
    sc.pp.filter_genes(ad_sp,min_counts = min_sp_counts)




    var_sc = ad_sc.var_names
    var_sp = ad_sp.var_names
    inter = var_sc.intersection(var_sp)

    if gene_list is not None:
        inter = pd.Index(gene_list).intersection(inter)

    ad_sc = ad_sc[:,inter]
    ad_sp = ad_sp[:,inter]

    ad_sc.X = ad_sc.to_df().values
    ad_sp.X = ad_sp.to_df().values

    return ad_sc,ad_sp


# In[54]:


def get_nb_params(adata, label_col, return_labels = False):

    _,n_genes = adata.shape
    labels = adata.obs[label_col].values.astype(str)
    uni_labels = np.unique(labels)
    n_labels = len(uni_labels)

    R = np.zeros((n_genes, n_labels))
    R_raw = np.zeros_like(R)
    P = np.zeros_like(R)

    X = adata.to_df().values

    for k,lab in enumerate(uni_labels):

        X_l = X[labels == lab]

        std2 = np.var(X_l, ddof = 1, axis = 0) + 1e-8
        mean = np.mean(X_l, axis=0) + 1e-8

        a = std2
        b = mean
        c = -mean

        p = np.clip((-b + np.sqrt(b**2 - 4*a*c)) / (2*a), 1e-8,1 - 1e-8)
        r = mean * p / (1 - p)

        R_raw[:,k] = r

        s = X_l.sum(axis=0, keepdims = True)

        r = np.divide(r, s, where = s.flatten() > 0)

        R[:,k] = r
        P[:,k] = p

    P = np.sum(P *R, axis =1) / np.sum(R,axis=1)

    R = pd.DataFrame(R, index = adata.var_names, columns = uni_labels)
    P = pd.DataFrame(P, index = adata.var_names)

    if return_labels:
        return R,P,uni_labels
    return R,P


# In[55]:


import torch
import torch.nn as nn
import pytorch_lightning as pl
from anndata import AnnData
from torch.utils.data import DataLoader, Dataset

# Custom Dataset to handle AnnData objects
class AnnDataDataset(Dataset):
    def __init__(self, adata: ad.AnnData, layer: str | None = None):
        self.X = t.tensor(adata.to_df(layer = layer).values)
        self.N = adata.shape[0]
        self.G = adata.shape[1]  

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:],idx


# In[56]:


# PyTorch Lightning model
from torch.distributions.negative_binomial import NegativeBinomial as NB

class RapidScope(pl.LightningModule):
    def __init__(self, r: np.ndarray, p : np.ndarray, n_spots: int , n_types : int, n_genes: int):
        super(RapidScope, self).__init__()

        self.w = nn.Parameter(t.ones(n_spots,n_types) / n_types,requires_grad=True) # n_spots x n_types
        self.nu = nn.Parameter(t.ones(1,n_genes),requires_grad=True) #1 x n_genes

        self.eta = nn.Parameter(t.zeros(1, n_genes))
        self.alpha = nn.Parameter(t.ones(1))

        self.r_sc = nn.Parameter(t.tensor(r.T.astype(np.float32)),requires_grad=False) #n_types x n_genes
        self.p_sc = nn.Parameter(t.tensor(p.T.astype(np.float32)),requires_grad = False) #n_types x n_genes

        self.smx = nn.Softmax(dim=1)
        self.spl = nn.Softplus()


    def get_props(self,):
        with t.no_grad():
            props = self.spl(self.w)
            props = props / props.sum(axis=1, keepdims = True)
        props = props.detach().cpu().numpy()
        return props        

    def forward(self, x, idx):
        beta = self.spl(self.nu) # 1 x n_genes
        v = self.spl(self.w[idx,:]) #n_spots x n_types
        eps = self.spl(self.eta)
        gamma = self.spl(self.alpha)

        r_sp = beta * t.mm(v, self.r_sc) + gamma * eps # n_genes x (n_spots, n_genes)
        p_sp = self.p_sc
        return r_sp,p_sp

    def loss(self, x, r, p):
        log_prob = NB(total_count=r, probs=p).log_prob(x)
        return -log_prob.sum()

    def training_step(self, batch, batch_idx):
        x, idx = batch
        r,p = self(x,idx)
        loss = self.loss(x,r,p)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# In[57]:


def run(ad_sc,ad_sp : ad.AnnData | List[ad.AnnData], label_col, batch_size: int = 1024,num_epochs: int = 500, **kwargs):

    if isinstance(ad_sp, list):
        _ad_sp = ad.concat(ad_sp,join='outer')
    else:
        _ad_sp = ad_sp.copy()

    _ad_sc, _ad_sp = preprocess(ad_sc.copy(), _ad_sp, label_col = label_col)
    R,P,labels = get_nb_params(_ad_sc,label_col = label_col, return_labels = True)

    dataset = AnnDataDataset(_ad_sp)


    model = RapidScope(R.values,
                       P.values,
                       n_spots = dataset.N,
                       n_genes = dataset.G,
                       n_types = R.shape[1])

    dataloader = DataLoader(dataset=dataset, batch_size= batch_size, shuffle=True)

    trainer = pl.Trainer(max_epochs = num_epochs,
                         accelerator= 'cuda' if t.cuda.is_available() else 'cpu')

    trainer.fit(model, dataloader)

    props = model.get_props()

    props = pd.DataFrame(props, index = _ad_sp.obs_names, columns = labels)

    return props



# In[58]:


props = run(ad_sc, ad_sp, label_col = label_col)


# In[59]:


col_names = props.columns


# In[60]:


if col_names[0] in ad_sp.obs.columns:
    ad_sp.obs.drop(col_names, inplace = True,axis=1)


# In[61]:


obs_new = pd.concat((ad_sp.obs, props,),axis =1)
obs_old = ad_sp.obs.copy()


# In[62]:


ad_sp.obs = obs_new


# In[63]:


sc.pl.spatial(ad_sp, color = col_names, spot_size =200)

