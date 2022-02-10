
import pandas as pd
import numpy as np
import os, pickle , time
from tqdm import tqdm
# from torchtext.legacy import data

import spacy
from spacy.vocab import Vocab
import functools




class LangModel:
	"""
	This classes Read a new copus, do the preprocess and save the data as Index.
	Save the vocabulary
	"""
	def __init__(self,max_s1 = 11,max_s2 = 6):
		self.nlp = spacy.load("en_core_web_md")
		self.dumbTok = self.nlp('sajhfauhsd')
		self.idx = 1
		self.str2idx = {}
		self.idx2str = {}
		self.idx2emb = []
		
		self.idx2str[0] = self.dumbTok.text
		self.idx2emb.append(self.dumbTok.vector)

		self.data = {}
		self.dataProcessed = {}
		self.maxTokens = {}
		self.dataPreProcess = {}
		
		self.dataPreProcess['S1'] = []
		self.dataPreProcess['S2'] = []

		self.maxTokens['S1'] = max_s1
		self.maxTokens['S2'] = max_s2
		
		self.labels = {'contradiction':0, 'entailment':1, 'neutral':2}

	def label_encod(self,lab):
		return self.labels[lab]

	def get_data(self,path):
		start = time.time()
		print('Start reading data',flush = True)
		for stage in ['train','dev','test']:
			# load dataset
			df = pd.read_csv(os.path.join(path,f'snli_1.0_{stage}.txt'), sep='\t')
			# Get neccesary columns
			df = df[['gold_label', 'sentence1', 'sentence2']]
			df.dropna( inplace=True)
			df = df[df['gold_label'] != '-']
			
			# Take small dataset
			# if stage== 'train':
			# 	df = df[:1000]
			# else:
			# 	df = df[:100]
			# 	df = df[:100]
	
			#get the lis of sentences:
			s1 = [i for i in df['sentence1'].to_list() if type(i) ==str]
			s2 = [i for i in df['sentence2'].to_list() if type(i) == str]
			lab = df['gold_label'].apply(self.label_encod).to_numpy()
			assert len(s1) == len(s2)
			self.data[stage] = (s1,s2,lab)

		sec = (time.time() - start)/60
		print("Data readed in ", sec, ' seg')

	def dataProcess(self):
		for stage in ['train','dev','test']:
			start = time.time()
			print(f'Start processing data {stage}',flush = True)
			sentence_pair = []
			s1, s2, lab = self.data[stage]
			assert len(s1) ==len(s2)

			s1 = self.sentence_cleaning(s1)
			s2 = self.sentence_cleaning(s2)

			funcS1 = functools.partial(self.uniform_shape, max_=self.maxTokens['S1'])
			funcS2 = functools.partial(self.uniform_shape, max_=self.maxTokens['S2'])
			
			sentence_1 = np.array(list(map(funcS1,s1)))
			sentence_2 = np.array(list(map(funcS2,s2)))

			self.dataProcessed[stage] = (sentence_1,sentence_2,lab)
			
		sec = (time.time() - start)/60
		print("Data Processed in ", sec, ' min')
		return None

	def sentence_cleaning(self, sentences):
		preProcess = []
		for i in tqdm(range(len(sentences))):
			sentence = sentences[i]
			sentence = sentence.lower()
			tokens = self.nlp(sentence)
			tokens = [tok for tok in tokens if (tok.is_stop == False)]
			tokens = [tok for tok in tokens if (tok.is_punct == False)]
			s = self.buildVocab(tokens)
			preProcess.append(s)
		return preProcess

	def buildVocab(self,tokens):
		final = []
		for word in tokens:
			if word.text not in self.str2idx.keys():
					self.str2idx[word.text] = self.idx
					self.idx2str[self.idx] = word.text
					self.idx2emb.append(word.vector)
					final.append(self.idx)
					self.idx +=1
			else:
				final.append(self.str2idx[word.text])
		return final

	def uniform_shape(self,sentence,max_):
		if max_ < len(sentence):
			sentence = sentence[:max_]
		else:
			sentence = sentence + [0]*(max_ - len(sentence))
		return sentence
		#return list(map(lambda x: trans[x],sentence))

	def save_processed(self,save_path):
		print('saving files...', flush = True)
		for stage in ['train','dev','test']:
			outfile = os.path.join(save_path,f'{stage}_idx')
			s1,s2,lab = self.dataProcessed[stage]
			np.savez(outfile, data_s1 = s1,data_s2 = s2,label = lab)
		
		outfile = os.path.join(save_path, f'Vocab')
		np.savez(outfile, Vocab = self.idx2emb)
		
		# outfile = os.path.join(save_path, f'Vocab.pkl')
		# with open(outfile, 'wb') as handle:
		# 	pickle.dump(self.idx2emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('all done')

# def analyse_data(path,stage):
# 	"""na verdade, eu deveria analisar o numero de tokens, e não das palavras em si"""
# 	df = pd.read_csv(os.path.join(path, f'snli_1.0_{stage}.txt'), sep='\t')
# 	df = df[['sentence1', 'sentence2']]
# 	nlp2 = spacy.load("en_core_web_md")
# 	def myFunc(stringg):
# 		sentence = stringg.lower()
# 		tokens = nlp2(sentence)
# 		tokens = [tok for tok in tokens if (tok.is_stop == False)]
# 		tokens = [tok for tok in tokens if (tok.is_punct == False)]
# 		return len(tokens)
# 	ls1 = df['sentence1'].apply(myFunc)
# 	ls2 = df['sentence2'].apply(myFunc)
# 	s1_len = int(np.percentile(ls1,90)) +1
# 	s2_len = int(np.percentile(ls2,90)) +1
# 	return s1_len,s2_len
# else:
# for i in tqdm(range(len(s1))):
# 	self.sentence_cleaning_train(s1[i], 'S1')
# 	self.sentence_cleaning_train(s2[i], 'S2')
# assert len(self.maxTokens['S1']) == len(s1)
# assert len(self.maxTokens['S2']) == len(s1)
# self.maxTokens['S1'] = int(np.percentile(self.maxTokens['S1'], 90))
# self.maxTokens['S2'] = int(np.percentile(self.maxTokens['S2'], 90))

# path = "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\snli_1.0\\"
# print(analyse_data(path ,'train'))