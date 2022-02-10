import numpy as np
import pandas as pd
import os,time
import spacy
from tqdm import tqdm


class macmorphoProcessing:
	"""
	This classes Read a new copus, do the preprocess and save the data as Index.
	"""
	
	def __init__(self, maxT=3):
		self.nlp = spacy.load("pt_core_news_md")
		self.dumbTok = self.nlp('sajhfauhsd')
		self.idx_w = 1
		self.str2idx = {}
		self.idx2str = {}
		self.idx2emb = {}
		self.idx_tag = 0
		self.str2idx = {}
		self.idx2str = {}
		self.idx2str[0] = self.dumbTok.text
		self.idx2emb[0] = self.dumbTok.vector
		self.data = {}
		self.maxTokens = maxT
		self.classes = {}
	
	def extracTAG(self, sentence):
		def str2tuple(s, sep="/"):
			loc = s.rfind(sep)
			if loc >= 0:
				return (s[:loc], s[loc + len(sep):].upper())

		res = {}
		res['words'] = []
		res['tags'] = []
		for word in sentence.split(' '):
			aux = str2tuple(word, sep='_')
			res['words'].append(aux[0].lower())
			res['tags'].append(aux[1])
			#assert len()
		return res
	
	def get_data(self, path):
		for stage in ['train','dev','test']:
			start = time.time()
			print('Start reading data', flush=True)
			
			# load dataset
			df = pd.read_csv(os.path.join(path, f'macmorpho-{stage}.txt'), sep='\n', header=None)
			#max_ = 100
			data = df.iloc[:,0].apply(self.extracTAG)
			if stage =='train':
				words_idx, tags =self.buildVocab_train(data)
			else:
				words_idx, tags = self.buildVocab_val(data)
			
			#self.words = self.get_inp_emb(words_idx,self.idx2emb)
			
			save_path = "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\pos_tagging\\"
			outfile = os.path.join(save_path, f'final_POS_tagging_{stage}')
			np.savez(outfile, words=words_idx, tags=tags)

		outfile = os.path.join(save_path, f'Vocab_idx2emb.pkl')
		with open(outfile, 'wb') as handle:
			pickle.dump(self.idx2emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
		outfile = os.path.join(save_path, f'Classes_idx2emb.pkl')
		with open(outfile, 'wb') as handle:
			pickle.dump(self.classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

		min = (time.time() - start) / 60
		print("Data readed in ", min, ' min')
		#return self.words,self.tags
		return None,None


	def buildVocab_val(self, data):
		final_w = []
		final_tag = []
		for sentence in tqdm(data):
			aux_w = []
			aux_tag = []
			for i, w in enumerate(sentence['words']):
				# if the thag has nob benn seen

				# if the word has not benn senn, add to vocab
				if w not in self.str2idx.keys():
					self.str2idx[w] = self.idx_w
					self.idx2str[self.idx_w] = w
					self.idx2emb[self.idx_w] = self.nlp(w).vector
					aux_w.append(self.idx_w)
					self.idx_w += 1
				# otherside, just add the acording index
				else:
					aux_w.append(self.str2idx[w])

				if i > 1:
					if sentence['tags'][i-1] not in self.classes.keys():
						continue
						
					else:
						tag_i = self.classes[sentence['tags'][i-1]]

					final_w.append([aux_w[i-2],aux_w[i-1],aux_w[i]])
					final_tag.append(tag_i)

		return np.array(final_w), np.array(final_tag)

	def buildVocab_train(self,data):
		final_w = []
		final_tag = []
		for sentence in tqdm(data):
			aux_w = []
			aux_tag = []
			for i,w  in enumerate(sentence['words']):

				# if the word has not benn senn, add to vocab
				if w not in self.str2idx.keys():
					self.str2idx[w] = self.idx_w
					self.idx2str[self.idx_w] = w
					self.idx2emb[self.idx_w] = self.nlp(w).vector
					aux_w.append(self.idx_w)
					self.idx_w += 1
				#otherside, just add the acording index
				else:
					aux_w.append(self.str2idx[w])

				if i > 1:
					# if the thag has nob benn seen
					if sentence['tags'][i-1] not in self.classes.keys():
						self.classes[sentence['tags'][i-1]] = self.idx_tag
						tag_i = self.classes[sentence['tags'][i-1]]
						self.idx_tag +=1
					else:
						tag_i = self.classes[sentence['tags'][i-1]]

					final_w.append([aux_w[i-2],aux_w[i-1],aux_w[i]])
					final_tag.append(tag_i)
		
		return np.array(final_w), np.array(final_tag)
	
	def get_inp_emb(self,sentIdx,translate):
		dat = [list(map(lambda x: translate[x], sentIdx[:, i])) for i in range(sentIdx.shape[-1])]
		return np.stack(dat, axis=1)
	


# TO RUN :

# if __name__ == "__main__":
# 	path_file = "C:\\Users\\gcram\\Documents\\Datasets\\NLP\\pos_tagging\\"
# 	dp = macmorphoProcessing()
# 	dp.get_data(path_file)