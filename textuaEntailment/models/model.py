from torch import nn
import torch


class lstmModel(nn.Module):
	def __init__(self,
	             n_classes = 3,
	             stacked_layers = 2,
	             hidden_size = 64,
	             embedding_dim = 300,
	             batch_size = 512,
	             seq_len1 = 11,
	             seq_len2 = 6):
		super(lstmModel, self).__init__()
		
		self.hidden_size = hidden_size
		self.bidirectional = True
		self.n_classes = n_classes
		self.stacked_layers = stacked_layers
		self.embedding_dim = embedding_dim
		self.seq_len1 = seq_len1
		self.seq_len2 = seq_len2
		self.batch_size = batch_size

		# self.vocab_size = vocab_size
		# num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
		# self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		# self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
		# self.embedding.weight.requires_grad = True
		
	def build(self):
		self.lstm_hypotheses  = nn.LSTM(input_size=self.embedding_dim,
		                    hidden_size=self.hidden_size,
		                    num_layers=self.stacked_layers,
		                    batch_first=True,
		                    dropout=0.2,
		                    bidirectional=self.bidirectional)
		self.lstm_evidences  = nn.LSTM(input_size=self.embedding_dim,
		                    hidden_size=self.hidden_size,
		                    num_layers=self.stacked_layers,
		                    batch_first=True,
		                    dropout=0.2,
		                    bidirectional=self.bidirectional)
		
		self.lstm_agg  = nn.LSTM(input_size= 2 * self.hidden_size,
						hidden_size=self.hidden_size,
						num_layers=self.stacked_layers,
						batch_first=True,
						dropout=0.2,
						bidirectional=self.bidirectional)

		self.FC = nn.Sequential(
			nn.Flatten(), # 4352 ?
			nn.Linear(2 * (self.seq_len1 +self.seq_len2)  * self.hidden_size, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Linear(32, self.n_classes),
			nn.Sigmoid()
		)
	
	
	def forward(self, input):
		hypotheses,evidences = input[0],input[1]

		# h0 = torch.zeros(self.stacked_layers * 2 ,
		#                  self.batch_size,
		#                  self.hidden_size).to('cuda') # 2 for bidirection
		#
		# c0 = torch.zeros(self.stacked_layers * 2 ,
		#                  self.batch_size,
		#                  self.hidden_size).to('cuda')
		#
		# hidden = self.init_hidden(batch_size)
		
		#hypotheses, (hH, ch) = self.lstm_hypotheses(hypotheses, (h0, c0))
		hypotheses, (hH, ch) = self.lstm_hypotheses(hypotheses)
		evidences, (hE, cE) = self.lstm_evidences(evidences, (hH, ch))
		comb_outputs = torch.cat( (hypotheses, evidences),1)
		#comb_hidden = torch.cat((hiddenH,hiddenE),-1)
		out, (hidden, _) = self.lstm_agg(comb_outputs,(hE, cE))
		probs = self.FC(out)
		return probs
	
# from torchsummary import summary
# m = lstmModel()
# m.build()
# m = m.to('cuda')
# summary(m, ((1000,7,96),(1000,7,96)), device='cuda')