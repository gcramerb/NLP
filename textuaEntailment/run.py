#from torchtext import data
from data.dataProcessing import LangModel

from data.DataClass import MyDataModule
from torch.utils.data import Dataset,DataLoader
from trainer import NLItrainer
from pytorch_lightning import Trainer
from trainer import NLItrainer

import sys, argparse,os,glob

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--saveModel', action='store_true')
args = parser.parse_args()

if args.slurm:
	proces_file = "/storage/datasets/sensors/guilherme/"

else:
	path_file =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_1.0\\"
	proces_file =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_test\\"


def process():
	dataProcessing = LangModel(stage = 'train')
	dataProcessing.get_data(path_file)
	seq_len1, seq_len2 = dataProcessing.dataProcess()
	dataProcessing.save_processed(proces_file)

	dataProcessing = LangModel(stage='dev',max_s1 = seq_len1,max_s2= seq_len2)
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(proces_file)
	#
	dataProcessing = LangModel(stage = 'test',max_s1 = seq_len1,max_s2= seq_len2)
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(proces_file)
	return seq_len1,seq_len2
	


if __name__ == '__main__':
	#seq_len1,seq_len2 = process()
	#print('TAMANHOS finais: ',seq_len1,'  ',seq_len2,'\n\n')
	model = NLItrainer(seq_len1 = 11,seq_len2= 6)
	dm = MyDataModule(batch_size = 512,path = proces_file)
	dm._setup('train')
	dm._setup('dev')
	dm._setup('test')

	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=20,
	                  logger=None,
	                  progress_bar_refresh_rate=1,
	                  callbacks=[])
	trainer.fit(model,datamodule = dm)
	trainer.test(dataloaders=dm.test_dataloader())

