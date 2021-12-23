#from torchtext import data
from data.dataProcessing import LangModel
from data.DataClass import MyDataModule
from torch.utils.data import Dataset,DataLoader
# from model import lstmModel
# from trainer import NLItrainer


path_file =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_1.0\\"
out_path =  "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_processed\\"


def process():
	dataProcessing = LangModel(stage = 'train')
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(out_path)
	
	dataProcessing = LangModel(stage = 'dev')
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(out_path)
	
	
	dataProcessing = LangModel(stage = 'test')
	dataProcessing.get_data(path_file)
	dataProcessing.dataProcess()
	dataProcessing.save_processed(out_path)

#process()
dm = MyDataModule(batch_size = 64,path = out_path)
dm.setup()
a = 10
# model = lstmModel()
#
# trainer = Trainer()
# trainer.fit(model, train_dataloader, val_dataloader)