
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
from ..distribution import _multinomial as st
import logging
logger = logging.getLogger(__name__)

def train_vae(etm,data,epochs,l_rate):
	logger.info('Starting training....')
	opt = torch.optim.Adam(etm.parameters(),lr=l_rate)
	loss_values = []
	data_size = data.dataset.shape[0]
	for epoch in range(epochs):
		loss = 0
		loss_ll = 0
		loss_kl = 0
		loss_klb = 0
		for x,y in data:
			opt.zero_grad()
			m,v,bm,bv,theta,beta,zz = etm(x)
			alpha = torch.exp(torch.clamp(torch.mm(theta,beta),-10,10))
			loglikloss = st.multi_dir_log_likelihood(x,alpha)
			kl = st.kl_loss(m,v)
			klb = st.kl_loss(bm,bv)

			train_loss = torch.mean(kl -loglikloss).add(torch.sum(klb)/data_size)
			train_loss.backward()

			opt.step()

			ll_l = torch.mean(loglikloss).to('cpu')
			kl_l = torch.mean(kl).to('cpu')
			klb_l = torch.sum(klb).to('cpu')

			loss += train_loss.item()
			loss_ll += ll_l.item()
			loss_kl += kl_l.item()
			loss_klb += klb_l.item()/data_size

		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss/len(data)))

		loss_values.append([loss/len(data),loss_ll/len(data),loss_kl/len(data),loss_klb/len(data)])


	return loss_values

def train_simple_flow_model(model, dataloader, epochs, lr, device):

	optimizer = optim.Adam(model.parameters(), lr=lr)
	loss_values = []
	data_size = dataloader.dataset.shape[0]
	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		eloss = 0
		for batch in dataloader:
	  
			z1 = batch[0].to(device)		
			z2 = batch[2].to(device)
   
			w, log_det = model(z1, z2)
			loss = 0.5 * torch.mean(w.pow(2)) - torch.mean(log_det)

			loss.backward()
			optimizer.step()

			eloss += loss.item()
   
		if (epoch + 1) % 10 == 0:
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {eloss/data_size:.4f}")

		loss_values.append([eloss/data_size])
  