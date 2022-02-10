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
	proces_file =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_processed\\"


def process():
	dataProcessing = LangModel(max_s1 = 11,max_s2 = 6)
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(proces_file)


if __name__ == '__main__':
	process()
	
	#print('TAMANHOS finais: ',seq_len1,'  ',seq_len2,'\n\n')
	model = NLItrainer(seq_len1 = 11,seq_len2= 6,path_emb = proces_file)
	dm = MyDataModule(batch_size = 64,path = proces_file)
	dm._setup()


	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=20,
	                  logger=None,
	                  progress_bar_refresh_rate=1,
	                  callbacks=[])
	trainer.fit(model,datamodule = dm)
	trainer.test(dataloaders=dm.test_dataloader())
	PATH_MODEL = "C:\\Users\\gcram\\Documents\\Github\\NLP\\saved\\texEnt_model.ckpt"
	trainer.save_checkpoint(PATH_MODEL)

