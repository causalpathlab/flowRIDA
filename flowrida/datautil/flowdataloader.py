import torch
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)



class MemDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x1_l, x2, x2_l):
        self.x1 = x1
        self.x1_l = x1_l
        self.x2 = x2
        self.x2_l = x2_l
        self.shape = x1.shape

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx],self.x1_l[idx],self.x2[idx],self.x2_l[idx]

    	
def nn_load_data_flow(x1,x1_l,x2,x2_l,batch_size):

    dataset = MemDataset(
			x1,
			x1_l,
			x2,
			x2_l			
		)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    	