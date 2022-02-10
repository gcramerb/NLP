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
import sys, os, argparse
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
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		
		# networks
		self.model = lstmModel(seq_len1 = seq_len1,seq_len2 = seq_len2)
		self.model.build()
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
	
	# def training_epoch_end(self, output):
	# 	metrics = {}
	# 	keys_ = output[0].keys()
	# 	for k in keys_:
	# 		metrics[k] = torch.mean(torch.stack([i[k] for i in output])
	# 	for k, v in metrics.items():
	# 		self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	

	
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










class LitMNIST(LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss


def train(model, train_loader, val_loader, criterion, optimizer):
	total_step = len(train_loader)
	
	for epoch in range(EPOCHS):
		start = time.time()
		model.train()
		total_train_loss = 0
		total_train_acc = 0
		for val in train_loader:
			sentence_pairs, labels = map(list, zip(*val))
			
			premise_seq = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
			hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
			batch = len(premise_seq)
			
			premise_len = list(map(len, premise_seq))
			hypothesis_len = list(map(len, hypothesis_seq))
			
			temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
			premise_seq = temp[:batch, :]
			hypothesis_seq = temp[batch:, :]
			labels = torch.tensor(labels).long().to(device)
			
			model.zero_grad()
			prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len)
			
			loss = criterion(prediction, labels)
			acc = multi_acc(prediction, labels)
			
			loss.backward()
			optimizer.step()
			
			total_train_loss += loss.item()
			total_train_acc += acc.item()
		
		train_acc = total_train_acc / len(train_loader)
		train_loss = total_train_loss / len(train_loader)
		model.eval()
		total_val_acc = 0
		total_val_loss = 0
		with torch.no_grad():
			for val in val_loader:
				sentence_pairs, labels = map(list, zip(*val))
				
				premise_seq = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
				hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
				batch = len(premise_seq)
				
				premise_len = list(map(len, premise_seq))
				hypothesis_len = list(map(len, hypothesis_seq))
				
				temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
				premise_seq = temp[:batch, :]
				hypothesis_seq = temp[batch:, :]
				
				premise_seq = premise_seq.to(device)
				hypothesis_seq = hypothesis_seq.to(device)
				labels = torch.tensor(labels).long().to(device)
				
				model.zero_grad()
				prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len)
				
				loss = criterion(prediction, labels)
				acc = multi_acc(prediction, labels)
				
				total_val_loss += loss.item()
				total_val_acc += acc.item()
		
		val_acc = total_val_acc / len(val_loader)
		val_loss = total_val_loss / len(val_loader)
		
		end = time.time()
		hours, rem = divmod(end - start, 3600)
		minutes, seconds = divmod(rem, 60)
		print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
		print(
			f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
		torch.cuda.empty_cache()
		
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)