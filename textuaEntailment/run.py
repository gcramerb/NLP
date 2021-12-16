#from torchtext import data
from data.dataProcessing import LangModel
from data.DataClass import MyDataModule
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


dm = MyDataModule(64,out_path)
dm.setup()

# model = lstmModel()
#
# trainer = Trainer()
# trainer.fit(model, train_dataloader, val_dataloader)