from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset
import os
import numpy as np
class myDataset(Dataset):
	def __init__(self ,data):
		self.myEmb = myEmb()
		
		self.sentences, self.label = self.dataProcess(data)

	def __len__(self):
		return len(self.sentence)
	
	def __getitem__(self, index):
		return self.sentence[index], self.labels[index]


class MyDataModule(LightningDataModule):
	def __init__(self ,batch_size, path):
		super().__init__()
		self.batch_size = batch_size
		self.path_file = path
	

	def setup(self):
		for stage in ['train','dev','test']:
			outfile = os.path.join(self.path_file,f'{stage}_idx.npy')
			idx =  np.load(outfile)
			outfile = os.path.join(self.path_file, f'{stage}_idx2emb.pkl')
			with open(outfile, 'rb') as handle:
				idx2emb = pickle.load(handle)

		
	
	def train_dataloader(self):
		return DataLoader(self.train, batch_size=self.batch_size)
	def val_dataloader(self):
		return DataLoader(self.dev, batch_size=self.batch_size)
	def test_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch_size)