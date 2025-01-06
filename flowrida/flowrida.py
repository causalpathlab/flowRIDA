from . import datautil as dutil
from . import model as md 
import torch
import logging
import pandas as pd

class flowrida(object):
    def __init__(self, data, wdir,tag):
        self.data = data
        self.wdir = wdir
        self.tag = tag
        logging.basicConfig(filename=self.wdir+'results/flowrida_train_'+self.tag+'.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
  
    def set_nn_params(self,params: dict):
        self.nn_params = params
    
    def set_flow_nn_params(self,params: dict):
        self.flow_nn_params = params
    
    def train(self):

        logging.info('Starting training...')

        logging.info(self.nn_params)
  
        flowrida_model = md.ETM(self.nn_params['input_dim'], self.nn_params['latent_dim'], self.nn_params['layers']).to(self.nn_params['device'])
  
        logging.info(flowrida_model)
  
        data =  dutil.nn_load_data(self.data,self.nn_params['device'],self.nn_params['batch_size'])

        loss = md.train(flowrida_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'])

        torch.save(flowrida_model.state_dict(),self.wdir+'results/flowrida_'+self.tag+'.model')
        df_loss = pd.DataFrame(loss,columns=['loss','loglik','kl','klb'])
        df_loss.to_csv(self.wdir+'results/flowrida_train_loss_'+self.tag+'.txt.gz',index=False,compression='gzip',header=True)
        logging.info('Completed training...model saved in results/flowrida_'+self.tag+'.model')

    def eval(self,device,eval_batch_size):

        logging.info('Starting eval...')

        logging.info(self.nn_params)
  
        flowrida_model = md.ETM(self.nn_params['input_dim'], self.nn_params['latent_dim'], self.nn_params['layers']).to(self.nn_params['device'])
  
        flowrida_model.load_state_dict(torch.load(self.wdir+'results/flowrida_'+self.tag+'.model', map_location=torch.device(device)))
  
        flowrida_model.eval()
  
        logging.info(flowrida_model)
  
        data =  dutil.nn_load_data(self.data,device,eval_batch_size)

        df_theta = pd.DataFrame()
        df_zz = pd.DataFrame()
  
        for xx,y in data:
            theta,beta,zz,ylabel = md.predict_batch_vae(flowrida_model,xx,y)
            df_theta = pd.concat([df_theta,pd.DataFrame(theta.cpu().detach().numpy(),index=ylabel)],axis=0)
            df_zz = pd.concat([df_zz,pd.DataFrame(zz.cpu().detach().numpy(),index=ylabel)],axis=0)

        df_beta = pd.DataFrame()
  
        for xx,y in data:
            theta,beta,zz,ylabel = md.predict_batch(flowrida_model,xx,y)
            df_beta = pd.DataFrame(beta.cpu().detach().numpy(),columns=self.data.var.index.values).T
            df_beta.columns = ['latent_'+str(i) for i in df_beta.columns]
            break
        
        df_theta = df_theta.loc[self.data.obs.index.values,:]
        df_zz = df_zz.loc[self.data.obs.index.values,:]
  
        df_theta.columns = ['latent_'+str(i) for i in df_theta.columns]
        df_zz.columns = ['latent_'+str(i) for i in df_zz.columns]

        self.data.obsm['theta']= df_theta
        self.data.obsm['latent']= df_zz
        self.data.varm['beta']= df_beta

    def train_flow(self,adata_c,adata_p):
        
        batch_size = self.flow_nn_params['batch_size']
        device = self.flow_nn_params['device']
        lr = self.flow_nn_params['learning_rate']
        epochs = self.flow_nn_params['epochs']

        dataloader = dutil.nn_load_data_flow(
            adata_c.obsm['latent'].values,\
            adata_c.obs.index.values, \
            adata_p.obsm['latent'].values,\
            adata_p.obs.index.values, \
            batch_size
        )

        model = md.SimpleFlowModel(self.flow_nn_params['input_dim'],self.flow_nn_params['condition_dim'],self.flow_nn_params['hidden_dim'],self.flow_nn_params['num_layers']).to(device)

        loss = md.train_simple_flow_model(model,dataloader,epochs,lr,device)
  
        torch.save(model.state_dict(),self.wdir+'results/flowrida_cinn.model')
        df_loss = pd.DataFrame(loss,columns=['loss'])
        df_loss.to_csv(self.wdir+'results/flowrida_cinn_train_loss.txt.gz',index=False,compression='gzip',header=True)
        logging.info('Completed training...model saved in results/flowrida_cinn.model')

    def eval_flow(self,adata_c,adata_p):
        
        batch_size = self.flow_nn_params['batch_size']
        device = self.flow_nn_params['device']

        dataloader = dutil.nn_load_data_flow(
            adata_c.obsm['latent'].values,\
            adata_c.obs.index.values, \
            adata_p.obsm['latent'].values,\
            adata_p.obs.index.values, \
            batch_size
        )

        model = md.SimpleFlowModel(self.flow_nn_params['input_dim'],self.flow_nn_params['condition_dim'],self.flow_nn_params['hidden_dim'],self.flow_nn_params['num_layers']).to(device)

        model.eval()
        
        df_z1 = pd.DataFrame()
        df_z2 = pd.DataFrame()
        for batch in dataloader:
            z1 = batch[0]
            z1_l = batch[1]
            z2 = batch[2]
            z2_l = batch[3]
            w, log_det = model(z1, z2)
            z1_recons = model.reverse(w, z2)
            df_z1 = pd.concat([df_z1,pd.DataFrame(z1_recons.cpu().detach().numpy(),index=z1_l)])
            df_z2 = pd.concat([df_z2,pd.DataFrame(w.cpu().detach().numpy(),index=z2_l)])
        
        df_z1 = df_z1.loc[adata_c.obs.index.values,:]
        df_z1.columns = ['recons_'+str(i) for i in df_z1.columns]
        adata_c.obsm['reconstructed'] = df_z1

        df_z2 = df_z2.loc[adata_p.obs.index.values,:]
        df_z2.columns = ['recons_'+str(i) for i in df_z2.columns]
        adata_p.obsm['reconstructed'] = df_z2
   

    def plot_loss(self,mode='vae'):
        import matplotlib.pylab as plt
        
        plt.rcParams.update({'font.size': 20})

        if mode=='vae':
            loss_f = self.wdir+'results/flowrida_train_loss_'+self.tag+'.txt.gz'
            fpath = self.wdir+'results/flowrida_train_loss_'+self.tag+'.png'
        elif mode == 'cinn':
            loss_f = self.wdir+'results/flowrida_cinn_train_loss.txt.gz'
            fpath = self.wdir+'results/flowrida_cinn_train_loss.png'
      
        pt_size=4.0
  
        data = pd.read_csv(loss_f)
        num_subplots = len(data.columns)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6*num_subplots), sharex=True)

        for i, column in enumerate(data.columns):
            data[[column]].plot(ax=axes[i], legend=None, linewidth=pt_size, marker='o') 
            axes[i].set_ylabel(column)
            axes[i].set_xlabel('epoch')
            axes[i].grid(False)

        plt.tight_layout()
        plt.savefig(fpath);plt.close()
  
def create_flowrida_object(adata,wdir,tag):
    return flowrida(adata,wdir,tag)