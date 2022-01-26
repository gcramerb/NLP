from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset,DataLoader
import os,pickle
import numpy as np

class myDataset(Dataset):
	"""
	Class that recives the already processed data.
	"""
	def __init__(self, s1,s2,lab):
		self.sentence1,self.sentence2,self.labels =s1,s2,lab
	def __len__(self):
		return len(self.labels)
	def __getitem__(self, index):
		return self.sentence1[index],self.sentence2[index], self.labels[index]


class MyDataModule(LightningDataModule):
	def __init__(self ,batch_size, path):
		super().__init__()
		self.batch_size = batch_size
		self.path_file = path
		self.dataset = {}
		self.dataset['train'] = None
		self.dataset['dev'] = None
		self.dataset['test'] = None
	
	def get_inp_emb(self,sentIdx,translate):
		dat = [list(map(lambda x: translate[x], sentIdx[:, i])) for i in range(sentIdx.shape[-1])]
		return np.stack(dat, axis=1)


	def _setup(self,stage = 'train'):
		outfile = os.path.join(self.path_file,f'{stage}_idx.npz')
		with np.load(outfile,allow_pickle=True) as tmp:
			data_s1 = tmp['data_s1']
			data_s2 = tmp['data_s2']
			label = tmp['label']
		outfile = os.path.join(self.path_file, f'{stage}_idx2emb.pkl')
		with open(outfile, 'rb') as handle:
			idx2emb = pickle.load(handle)
		sentence1= self.get_inp_emb(data_s1, idx2emb)
		sentence2 = self.get_inp_emb(data_s2, idx2emb)
		self.dataset[stage] = myDataset(sentence1, sentence2,label)


	def train_dataloader(self):
		return DataLoader(self.dataset['train'], batch_size=self.batch_size)
	def val_dataloader(self):
		return DataLoader(self.dataset['dev'], batch_size=self.batch_size)
	def test_dataloader(self):
		return DataLoader(self.dataset['test'], batch_size=self.batch_size)