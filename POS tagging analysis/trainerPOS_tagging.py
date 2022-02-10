
#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
import os,pickle
import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')


from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""


class lstmModelPOS(nn.Module):
	def __init__(self,
	             n_classes=25,
	             stacked_layers=2,
	             hidden_size=128,
	             embedding_dim=300,
	             batch_size=256,
	             seq_len=3):
		super(lstmModelPOS, self).__init__()
		
		self.hidden_size = hidden_size
		self.bidirectional = True
		self.n_classes = n_classes
		self.stacked_layers = stacked_layers
		self.embedding_dim = embedding_dim
		self.seq_len = seq_len
		self.batch_size = batch_size
	
	def build(self):
		self.lstm = nn.LSTM(input_size=self.embedding_dim,
		                    hidden_size=self.hidden_size,
		                    num_layers=self.stacked_layers,
		                    batch_first=True,
		                    dropout=0.2,
		                    bidirectional=self.bidirectional)
		
		self.Flat = nn.Flatten()
		self.FC = nn.Linear(self.hidden_size * 2*self.seq_len, self.n_classes)
		self.SM = nn.Softmax(dim=1)
		
		#self.dropout = nn.Dropout(dropout)
	
	def forward(self, words):
		out, (hH, ch) = self.lstm(words)
		out = self.Flat(out)
		tag_space = self.FC(out)
		tag_scores = self.SM(tag_space)
		return tag_scores

class POStag_net(LightningModule):
	
	def __init__(
			self,
			lr: float = 0.01,
			n_classes = 25,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		
		# networks
		self.model = lstmModelPOS(n_classes = n_classes)
		self.model.build()
		self.loss = torch.nn.CrossEntropyLoss()
	
	def forward(self, X):
		return self.model(X)
	
	def _shared_eval_step(self, batch, stage='val'):
		words, tags = batch[0], batch[1].long()
		pred = self.model(words)
		
		if stage == 'val':
			loss = self.loss(pred, tags)
			tags_ = tags.cpu().numpy().flatten()
			pred_ = np.argmax(pred.cpu().numpy(),axis = 1).flatten()
			acc = accuracy_score(tags_,pred_)
			metrics = {'val_acc': acc,
			           'val_loss': loss.detach()}
		
		elif stage == 'test':
			tags_ = tags.cpu().numpy().flatten()
			pred_ = np.argmax(pred.cpu().numpy(),axis = 1).flatten()
			acc = accuracy_score(tags_,pred_)
			metrics = {'test_acc': acc}
		return metrics
	
	def training_step(self, batch, batch_idx):
		words, tags = batch[0], batch[1].long()
		pred = self.model(words)
		loss = self.loss(pred, tags)
		tqdm_dict = {f"train_loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict})
		return output


	def validation_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch, stage='val')
		return metrics
	
	def validation_epoch_end(self, out):
		keys_ = out[0].keys()
		metrics = {}
		for k in keys_:
			val = [i[k] for i in out]
			if 'loss' in k:
				metrics[k] = torch.mean(torch.stack(val))
			else:
				metrics[k] = np.mean(val)
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	def test_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch, stage='test')
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics
	
	def predict(self,dl,stage ='test'):
		outcomes = {}
		with torch.no_grad():

			true_list = []
			pred_list = []

			for batch in dl:
				words, tags = batch[0], batch[1].long()
				pred = self.model(words)
				tags_ = tags.cpu().numpy().flatten()
				pred_ = np.argmax(pred.cpu().numpy(), axis=1).flatten()
				true_list.append(tags_)
				pred_list.append(pred_)

			outcomes[f'true_{stage}'] = np.concatenate(true_list, axis=0)
			outcomes[f'pred_{stage}'] = np.concatenate(pred_list, axis=0)
			return outcomes

	def configure_optimizers(self):
		opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
		return [opt]


class myDataset(Dataset):
	"""
	Class that recives the already processed data.
	"""
	
	def __init__(self, w, tag, vocab):
		self.w, self.tag = w, tag
		self.vocab = vocab
	
	def get_inp_emb(self, sentIdx):
		dat = [list(map(lambda x: self.vocab[x], sentIdx[:, i])) for i in range(sentIdx.shape[-1])]
		return np.concatenate(dat)
	
	def __len__(self):
		return len(self.w)
	
	def __getitem__(self, index):
		return self.get_inp_emb(self.w[[index], :]), self.tag[index]


class MyDataModule(LightningDataModule):
	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.dataset = {}
		self.dataset['train'] = None
		self.dataset['dev'] = None
		self.dataset['test'] = None
	
	def _setup(self, path_file):
		outfile = os.path.join(path_file, f'Vocab_idx2emb.pkl')
		with open(outfile, 'rb') as handle:
			vocab = pickle.load(handle)
		outfile = os.path.join(path_file, f'Classes_idx2emb.pkl')
		with open(outfile, 'rb') as handle:
			self.n_classes = pickle.load(handle)
		
		for stage in ['train', 'dev', 'test']:
			outfile = os.path.join(path_file, f'final_POS_tagging_{stage}.npz')
			with np.load(outfile, allow_pickle=True) as tmp:
				words_idx = tmp['words']
				tags = tmp['tags']
			self.dataset[stage] = myDataset(words_idx, tags, vocab)
	
	def train_dataloader(self):
		return DataLoader(self.dataset['train'],
		                  drop_last=True,
		                  batch_size=self.batch_size)
	
	def val_dataloader(self):
		return DataLoader(self.dataset['dev'],
		                  drop_last=True,
		                  batch_size=self.batch_size)
	
	def test_dataloader(self):
		return DataLoader(self.dataset['test'],
		                  drop_last=True,
		                  batch_size=self.batch_size)


#TO RUN :

# import numpy as np
#
# import os,pickle
# from torch.utils.data import Dataset,DataLoader
# from torch import nn
# import torch
# from pytorch_lightning import LightningDataModule, LightningModule
# from pytorch_lightning import Trainer
# from trainerPOS_tagging import POStag_net,MyDataModule
#
# if __name__ == "__main__":
# 	path_file = "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\pos_tagging\\"
#
# 	dm = MyDataModule(batch_size=256)
# 	dm._setup(path_file)
#
# 	net = POStag_net(n_classes=len(dm.n_classes.keys()))
# 	trainer = Trainer(gpus=1,
# 	                  check_val_every_n_epoch=1,
# 	                  max_epochs=20,
# 	                  logger=None,
# 	                  progress_bar_refresh_rate=1,
# 	                  callbacks=[])
#
# 	trainer.fit(net, datamodule=dm)
# 	trainer.test(dataloaders=dm.test_dataloader())
#
# 	PATH_MODEL = "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\saved\\POStag_model.ckpt"
# 	trainer.save_checkpoint(PATH_MODEL)



