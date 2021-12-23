from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset,DataLoader
import os,pickle
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
	
	def get_inp_emb(self,idx,translate):
		s1,s2 = idx[:,0,:],idx[:,1,:]
		sentence1 = np.array([np.array(list(map(lambda x: translate[x],s))) for s in s1])
		sentence2 = np.array([np.array(list(map(lambda x: translate[x], s))) for s in s2])
		return (sentence1,sentence2)


	def setup(self):
		self.data = {}
		for stage in ['train','dev','test']:
			outfile = os.path.join(self.path_file,f'{stage}_idx.npz')
			with np.load(outfile) as tmp:
				data = tmp['data']
				label = tmp['label']
			outfile = os.path.join(self.path_file, f'{stage}_idx2emb.pkl')
			with open(outfile, 'rb') as handle:
				idx2emb = pickle.load(handle)
			self.data[stage] = self.get_inp_emb(data,idx2emb),label


	def train_dataloader(self):
		return DataLoader(self.data['train'], batch_size=self.batch_size)
	def val_dataloader(self):
		return DataLoader(self.data['dev'], batch_size=self.batch_size)
	def test_dataloader(self):
		return DataLoader(self.dat['test'], batch_size=self.batch_size)