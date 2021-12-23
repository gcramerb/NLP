from torch import nn


class lstmModel(nn.Module):
	def __init__(self, vocab_size,
	             n_classes = 3,
	             stacked_layers = 2,
	             hidden_size = 64,
	             embedding_dim = 96):
		super(lstmModel, self).__init__()
		
		self.hidden_size = hidden_size
		self.bidirectional = True
		self.n_classes = n_classes
		self.stacked_layers = stacked_layers
		self.embedding_dim = embedding_dim
		
		
		
		#
		# self.vocab_size = vocab_size
		# num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
		# self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		# self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
		# self.embedding.weight.requires_grad = True
		
	def build(self):

		self.lstm_hypotheses  = nn.LSTM(input_size=self.embedding_dim,
		                    hidden_size=self.hidden_size,
		                    num_layers=1,
		                    batch_first=True,
		                    dropout=0.2,
		                    bidirectional=bidirectional)
		
		self.lstm_evidences  = nn.LSTM(input_size=self.embedding_dim,
		                    hidden_size=self.hidden_size,
		                    num_layers=1,
		                    batch_first=True,
		                    dropout=0.2,
		                    bidirectional=bidirectional)
		
		self.lstm_agg  = nn.Sequential(
						nn.LSTM(input_size=self.embedding_dim,
						hidden_size=self.hidden_size,
						num_layers=1,
						batch_first=True,
						dropout=0.2,
						bidirectional=bidirectional),
						nn.ReLU(),
						nn.Dropout(p=0.2)
		)

		
		self.FC = nn.Sequential(
			nn.Linear(2 * 2 * hidden_size, 128),
			nn.Linear(128, 64),
			nn.Linear(64, 32),
			nn.Linear(32, self.n_classes)
		)
	
	
	def forward(self, input):
		premise = input[0]
		hypothesis = input[1]
		# batch_size = premise.size(0)
		#
		# h0 = torch.zeros(self.stacked_layers * 2 ,
		#                  batch_size,
		#                  self.hidden_size).to(device)  # 2 for bidirection
		#
		# c0 = torch.zeros(self.stacked_layers * 2 ,
		#                  batch_size,
		#                  self.hidden_size).to(device)
		
		# hidden = self.init_hidden(batch_size)
		
		premise, (hidden, _) = self.lstm_premise(premise)

		hypothesis, (hidden, _) = self.lstm_hypthesis(hypothesis)

		combined_outputs = torch.cat(
									(premise,
									 hypothesis
									 )
									 # torch.abs(premise - hypothesis),
									 # premise * hypothesis),
		                             # dim=2)
		out = self.lstm_agg(combined_outputs)
		probs = self.FC(out)
		return probs