# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
USE_CUDA = True


class Sclstm(nn.Module):
	def __init__(self, hidden_size, vocab_size, d_size, dropout=0.5):
		super(Sclstm, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.dropout = dropout
	
		self.w2h = nn.Linear(vocab_size, hidden_size*4)
		self.h2h = nn.Linear(hidden_size, hidden_size*4)

		self.w2h_r= nn.Linear(vocab_size, d_size)
		self.h2h_r= nn.Linear(hidden_size, d_size)

		self.dc = nn.Linear(d_size, hidden_size, bias=False)
		self.out = nn.Linear(hidden_size, vocab_size)


	def _step(self, input_t, last_hidden, last_cell, last_dt):
		'''
		* Do feedforward for one step *
		Args:
			input_t: (batch_size, 1, hidden_size)
			last_hidden: (batch_size, hidden_size)
			last_cell: (batch_size, hidden_size)
		Return:
			cell, hidden at this time step
		'''
		# get all gates
		input_t = input_t.squeeze(1)
		w2h = self.w2h(input_t) # (batch_size, hidden_size*5)
		w2h = torch.split(w2h, self.hidden_size, dim=1) # (batch_size, hidden_size) * 4
		h2h = self.h2h(last_hidden)
		h2h = torch.split(h2h, self.hidden_size, dim=1)

		gate_i = F.sigmoid(w2h[0] + h2h[0]) # (batch_size, hidden_size)
		gate_f = F.sigmoid(w2h[1] + h2h[1])
		gate_o = F.sigmoid(w2h[2] + h2h[2])

		# updata dt
		alpha = 0.5
		gate_r = F.sigmoid(self.w2h_r(input_t) + alpha * self.h2h_r(last_hidden))
		dt = gate_r * last_dt

		cell_hat = F.tanh(w2h[3] + h2h[3])
		cell = gate_f * last_cell + gate_i * cell_hat + self.dc(dt)
		hidden = gate_o * F.tanh(cell)

		return hidden, cell, dt

	
	def forward(self, input_seq, last_dt, dataset, gen=False):
		'''
		Args:
			input_seq: (batch_size, max_len, emb_size)
			dt: (batch_size, feat_size)
		Return:
			output_all: (batch_size, max_len, vocab_size)
		'''
		batch_size = input_seq.size(0)
		max_len = input_seq.size(1)
	
		output_all = Variable(torch.zeros(batch_size, max_len, self.vocab_size))
	
		# prepare init h and c
		last_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
		last_cell = Variable(torch.zeros(batch_size, self.hidden_size))
		if USE_CUDA:
			last_hidden = last_hidden.cuda()
			last_cell = last_cell.cuda()
			output_all = output_all.cuda()
	
		decoded_words = ['' for k in range(batch_size)]
		input_t = self.get_1hot_input(batch_size, dataset)
		for t in range(max_len):
			hidden, cell, dt = self._step(input_t, last_hidden, last_cell, last_dt)
			if not gen:
				hidden = F.dropout(hidden, p=self.dropout)
			output = self.out(hidden) # (batch_size, vocab_size)
			output_all[:, t, :] = output

			last_hidden, last_cell, last_dt = hidden, cell, dt
			previous_out = self.logits2words(output, decoded_words, dataset)

			input_t = previous_out if gen else input_seq[:, t, :]

		return output_all, decoded_words


	def get_1hot_input(self, batch_size, dataset):
		res = [[1 if index==dataset.word2index['SOS_token'] else 0 for index in range(self.vocab_size)] for b in range(batch_size)]
		res = Variable(torch.FloatTensor(res)) 
		if USE_CUDA:
			res = res.cuda()
		return res


	def logits2words(self, output, decoded_words, dataset):
		'''
		* Decode words from logits output at a time step AND put decoded words in final results*
		'''
		batch_size = output.size(0)
		topv, topi = F.softmax(output, dim=1).data.topk(1) # both (batch_size, 1)
		decoded_words_t = np.zeros((batch_size, self.vocab_size))
		for b in range(batch_size):
			idx = topi[b][0]
			word = dataset.index2word[idx]
			decoded_words[b] += (word + ' ')
			decoded_words_t[b][idx] = 1
		decoded_words_t = Variable(torch.from_numpy(decoded_words_t.astype(np.float32)))

		if USE_CUDA:
			decoded_words_t = decoded_words_t.cuda()

		return decoded_words_t 
	
