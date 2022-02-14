#
# EPOCHS = 25
# BATCH_SIZE = 32
# EMBEDDING_SIZE = 300
# VOCAB_SIZE = len(vocab.word2index)
# TARGET_SIZE = len(tag2idx)
# LEARNING_RATE = 0.005

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse,pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.model import lstmModel
# import geomloss

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""


class NLItrainer(LightningModule):
	
	def __init__(
			self,
			lr: float = 0.002,
			seq_len1: int = 11,
			seq_len2: int = 6,
			path_emb: str = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		if path_emb:
			emb_matrix = []
			file = os.path.join(path_emb, f'Vocab.npz')
			with np.load(file,allow_pickle=True) as tmp:
				emb_matrix = tmp['Vocab']


		# networks
		self.model = lstmModel(seq_len1 = seq_len1,seq_len2 = seq_len2)
		self.model.build(emb_matrix)
		self.loss = torch.nn.CrossEntropyLoss()
	
	def forward(self, X):
		return self.model(X)
	

	def _shared_eval_step(self, batch, stage='val'):
		sent1, sent2,label = batch[0], batch[1],batch[2].long()
		pred = self.model((sent1,sent2))
		if stage == 'val':
			loss = self.loss(pred, label)
			acc = accuracy_score(label.cpu().numpy(), np.argmax(pred.cpu().numpy(), axis=1))
			metrics = {'val_acc': acc,
			           'val_loss': loss.detach()}

		elif stage == 'test':
			acc = accuracy_score(label.cpu().numpy(), np.argmax(pred.cpu().numpy(), axis=1))
			metrics = {'test_acc': acc}
		return metrics
	
	def training_step(self, batch, batch_idx):
		
		sent1, sent2, label = batch[0], batch[1], batch[2].long()
		pred = self.model((sent1,sent2))
		loss = self.loss(pred,label)
		tqdm_dict = {f"train_loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict})
		
		return output
	
	def predict(self,dl,stage ='test'):
		outcomes = {}
		with torch.no_grad():

			true_list = []
			pred_list = []

			for batch in dl:
				sent1, sent2, label = batch[0], batch[1], batch[2].long()
				pred = self.model((sent1, sent2))
				true_ = label.cpu().numpy()
				pred_ = np.argmax(pred.cpu().numpy(), axis=1)
				true_list.append(true_)
				pred_list.append(pred_)

			outcomes[f'true_{stage}'] = np.concatenate(true_list, axis=0)
			outcomes[f'pred_{stage}'] = np.concatenate(pred_list, axis=0)
			return outcomes
	
	def validation_step(self, batch, batch_idx):
		# with torch.no_grad():
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

	def configure_optimizers(self):
		opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
		return [opt]
