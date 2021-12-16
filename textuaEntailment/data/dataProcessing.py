
import pandas as pd
import numpy as np
import os, pickle , time
# from torchtext.legacy import data


import spacy
from spacy.vocab import Vocab


SOS_token = 0
EOS_token = 1


class LangModel:
	"""
	This classes Read a new copus, do the preprocess and save the data as Index.
	"""
	def __init__(self,maxTokens = 7,stage = 'train'):
		self.nlp = spacy.load("en_core_web_sm")
		self.dumbTok = self.nlp("dskfodkfos")
		self.idx = 0
		self.str2idx = {}
		self.idx2str = {}
		self.idx2emb = {}
		self.data = {}
		self.maxTokens = maxTokens
		self.labels = {'contradiction':0, 'entailment':1, 'neutral':2}
		self.stage = stage

	def label_encod(self,lab):
		return self.labels[lab]

	def get_data(self,path):
		start = time.time()
		print('Start reading data',flush = True)
		# load dataset
		df = pd.read_csv(os.path.join(path,f'snli_1.0_{self.stage}.txt'), sep='\t')
		# df_dev = pd.read_csv(os.path.join(path,'snli_1.0_dev.txt'), sep='\t')
		# df_test = pd.read_csv(os.path.join(path,'snli_1.0_test.txt'), sep='\t')
		
		# Get neccesary columns
		df = df[['gold_label', 'sentence1', 'sentence2']]
		# df_dev = df_dev[['gold_label', 'sentence1', 'sentence2']]
		# df_test = df_test[['gold_label', 'sentence1', 'sentence2']]
		
		# Take small dataset
		if self.stage== 'train':
			df = df[:10000]
		else:
			df = df[:1000]
			df = df[:1000]
		
		df = df[df['gold_label'] != '-']
		# df_dev = df_dev[df_dev['gold_label'] != '-']
		# df_test = df_test[df_test['gold_label'] != '-']
		
		# Trim each sentence upto maximum length
		#train_s1= df_train['sentence1'].apply(sentence_cleaning).to_numpy()
		s1 = df['sentence1'].to_list()
		s2 = df['sentence2'].to_list()
		# dev_s1 = df_dev['sentence1'].to_list()
		# dev_s2 = df_dev['sentence2'].to_list()
		# test_s1 =  df_test['sentence1'].to_list()
		# test_s2 = df_test['sentence2'].to_list()
		
		
		lab = df['gold_label'].apply(self.label_encod).to_numpy()
		# lab_dev = df_dev['gold_label'].apply(self.label_encod).to_numpy()
		# lab_test = df_test['gold_label'].apply(self.label_encod).to_numpy()
		#
		self.data= (s1,s2,lab)
		# self.data['dev'] = (dev_s1,dev_s2,lab_dev)
		# self.data['test'] = (test_s1,test_s2,lab_test)
		
		sec = (time.time() - start)/60
		print("Data readed in ", sec, ' seg')


	def dataProcess(self):
		start = time.time()
		print(f'Start processing data {self.stage}',flush = True)
		sentence_pair = []
		s1, s2, lab = self.data
		for i in range(len(s1)):
			sentence_pair.append((self.sentence_cleaning(s1[i]),self.sentence_cleaning(s2[i])))
		self.dataProcessed = np.array(sentence_pair)

		sec = (time.time() - start)/60
		print("Data Processed in ", sec, ' seg')

	def sentence_cleaning(self, sentence):
		sentence = sentence.lower()
		tokens = self.nlp(sentence)
		tokens = [tok for tok in tokens if (tok.is_stop == False)]
		tokens = [tok for tok in tokens if (tok.is_punct == False)]
		sentenceIdx = self.addSentence(tokens)
		return np.array(sentenceIdx)
	
	def addSentence(self,tokens):
		final = []
		try:
			for k in range(self.maxTokens - len(tokens)):
				tokens.append(self.dumbTok)
			for i in range(self.maxTokens):
				word = tokens[i]
				if word.text not in self.str2idx.keys():
						self.str2idx[word.text] = self.idx
						self.idx2str[self.idx] = word.text
						self.idx2emb[self.idx] = word.vector
						self.idx +=1
				final.append(self.str2idx[word.text])
			return final
		except:
			a = 2

	def save_processed(self,save_path):
		print('saving files...', flush = True)
		outfile = os.path.join(save_path,f'{self.stage}_idx')
		np.save(outfile, self.dataProcessed)
		outfile = os.path.join(save_path, f'{self.stage}_idx2emb.pkl')
		with open(outfile, 'wb') as handle:
			pickle.dump(self.idx2emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('all done')
		#save the vocabulary Later(?)

