from . import dutil 
from . import model 
import torch
import logging
import gc
import pandas as pd
import numpy as np
import itertools

class gcan(object):
	def __init__(self, data: dutil.data.Dataset, wdir: str):
		self.data = data
		self.wdir = wdir
		logging.basicConfig(filename=self.wdir+'results/4_gcan_train.log',
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S')
				
	def set_nn_params(self,params: dict):
		self.nn_params = params
	
	def train(self):

		logging.info('Starting training...')

		logging.info(self.nn_params)
  
		gcan_model = model.ETM(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_rate'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		logging.info(gcan_model)
  
		data = dutil.nn_load_data_pairs(self.data,self.nn_params['device'],self.nn_params['batch_size'])

		loss = model.nn_attn.train(gcan_model,data,self.nn_params['epochs'],self.nn_params['lambda_loss'],self.nn_params['learning_rate'],self.nn_params['rare_ct_mode'],self.nn_params['num_clusters'],self.nn_params['rare_group_threshold'], self.nn_params['rare_group_weight'],self.nn_params['temperature_cl'])

		torch.save(gcan_model.state_dict(),self.wdir+'results/nn_gcan.model')
		pd.DataFrame(loss,columns=['ep_l','cl','el','el_attn_sc','el_attn_sp','el_cl_sc','el_cl_sp']).to_csv(self.wdir+'results/4_gcan_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in results/nn_gcan.model')
