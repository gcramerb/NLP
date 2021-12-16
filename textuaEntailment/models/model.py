from torch import nn


class lstmModel(nn.Module):
	def __init__(self, vocab_size,
	             target_size = 3,
	             weights_matrix,
	             stacked_layers = 2,
	             hidden_size = 64):
		super(lstmModel, self).__init__()
		
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		self.target_size = target_size
		self.stacked_layers = stacked_layers
		
		num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
		self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
		self.embedding.weight.requires_grad = True
		
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=self.stacked_layers,
		                    batch_first=True, dropout=0.2, bidirectional=bidirectional)
		
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.2)
		
		self.FC_concat1 = nn.Linear(2 * 2 * hidden_size if bidirectional else 2 * hidden_size, 128)
		self.FC_concat2 = nn.Linear(128, 64)
		self.FC_concat3 = nn.Linear(64, 32)
		
		for lin in [self.FC_concat1, self.FC_concat2]:
			nn.init.xavier_uniform_(lin.weight)
			nn.init.zeros_(lin.bias)
		
		self.output = nn.Linear(32, self.target_size)
		
		self.out = nn.Sequential(
			self.FC_concat1,
			self.relu,
			self.dropout,
			self.FC_concat2,
			self.relu,
			self.FC_concat3,
			self.relu,
			self.dropout,
			self.output
		)
	
	def forward_once(self, seq, hidden, seq_len):
		embedd_seq = self.embedding(seq)
		packed_seq = pack_padded_sequence(embedd_seq, lengths=seq_len, batch_first=True, enforce_sorted=False)
		output, (hidden, _) = self.lstm(packed_seq, hidden)
		return hidden
	
	def forward(self, input, premise_len, hypothesis_len):
		premise = input[0]
		hypothesis = input[1]
		batch_size = premise.size(0)
		
		h0 = torch.zeros(self.stacked_layers * 2 if self.bidirectional else self.stacked_layers, batch_size,
		                 self.hidden_size).to(device)  # 2 for bidirection
		c0 = torch.zeros(self.stacked_layers * 2 if self.bidirectional else self.stacked_layers, batch_size,
		                 self.hidden_size).to(device)
		
		# hidden = self.init_hidden(batch_size)
		
		premise = self.forward_once(premise, (h0, c0), premise_len)
		hypothesis = self.forward_once(hypothesis, (h0, c0), hypothesis_len)
		
		combined_outputs = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis),
		                             dim=2)
		
		return self.out(combined_outputs[-1])