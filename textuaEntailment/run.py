#from torchtext import data
from data.dataProcessing import LangModel
from data.DataClass import MyDataModule
from torch.utils.data import Dataset,DataLoader
from trainer import NLItrainer
from pytorch_lightning import Trainer
# from model import lstmModel
# from trainer import NLItrainer


path_file =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_1.0\\"
out_path =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_processed\\"


def process():
	dataProcessing = LangModel(stage = 'test')
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(out_path)

	dataProcessing = LangModel(stage='dev')
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(out_path)
	
	dataProcessing = LangModel(stage = 'train')
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(out_path)
	


#process()
model = NLItrainer()
dm = MyDataModule(batch_size = 64,path = out_path)
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
trainer.test(dataloaders=dm.test_dataloaders())

a = 10
# model = lstmModel()
#
# trainer = Trainer()
# trainer.fit(model, train_dataloader, val_dataloader)