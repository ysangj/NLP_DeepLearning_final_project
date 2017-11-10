import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import re
import random

import collections
import numpy as np

from torchtext import data
from torchtext import datasets
from model import EncoderRNN, DecoderRNN

device = 0 if(torch.cuda.is_available()) else -1

FR = data.Field(init_token='<sos>', eos_token='<eos>')
EN = data.Field(init_token='<sos>', eos_token='<eos>')

SOS_token = 2
EOS_token = 3
batch_size = 1 #switch batch_size to 3 or 5 when using train()
learning_rate=0.0001

train, val, test = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3

EN.build_vocab(train.src, min_freq=300)
FR.build_vocab(train.trg, min_freq=300)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=batch_size, device=device)  #set device as any number other than -1 in cuda envicornment

# define model
encoder = EncoderRNN(input_size = len(EN.vocab), hidden_size = 100)
decoder = DecoderRNN(hidden_size=100, output_size = len(FR.vocab))

# define loss criterion
criterion = nn.NLLLoss()
teacher_forcing_ratio = 0.5

# define optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)



train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=batch_size, device=device)  #set device as any number other than -1 in cuda envicornment

# define model
encoder = EncoderRNN(input_size = len(EN.vocab), hidden_size = 100)
decoder = DecoderRNN(hidden_size=100, output_size = len(FR.vocab))

# define loss criterion
criterion = nn.NLLLoss()
teacher_forcing_ratio = 0.5

# define optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)



def is_eos(topi, batch_size):
	eos_counter = 0
	for i in range(0, batch_size):
		if topi[i][0] == 3 or topi[i][0] == 1 :
			eos_counter += 1

	if eos_counter == batch_size:
		return True
	return False


def toy_train(src, trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
	loss = 0

	# encode
	encoder_hidden = encoder.init_hidden(batch_size)
	encoder_out, context = encoder(train_batch.src, encoder_hidden)
	#context = contexts[len(contexts)-1]

	# decode
	if torch.cuda.is_available():
	    decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*batch_size]))
	else:
	    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]))
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	if use_teacher_forcing:
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, batch_size)
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input = trg[trg_index].view(1, len(trg[trg_index]))
	else:
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, batch_size)
			#TODO: switch to sampling. For now, this is greedy
			topv, topi = decoder_output.data.topk(1)

			loss += criterion(decoder_output, trg[trg_index])
			decoder_input  = Variable(topi.view(1, len(topi)) )
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			# print(decoder_output)
			if is_eos(topi, batch_size):
				break

	decoder_optimizer.zero_grad()
	encoder_optimizer.zero_grad()

	loss.backward()
	decoder_optimizer.step()
	encoder_optimizer.step()
	trglength = len(trg)

	return loss.data[0]/ trglength




def train(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
	train_batch = next(iter(train_iter))
	src = train_batch.src
	trg = train_batch.trg
	
	loss = 0

	# encode
	encoder_hidden = encoder.init_hidden(batch_size)
	encoder_out, context = encoder(train_batch.src, encoder_hidden)

	# decode
	decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]))
	decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	if use_teacher_forcing:
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, batch_size)
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input = trg[trg_index].view(1, len(trg[trg_index]))
			#decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
	else:
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, batch_size)
			#TODO: switch to sampling. For now, this is greedy
			topv, topi = decoder_output.data.topk(1)

			loss += criterion(decoder_output, trg[trg_index])
			decoder_input  = Variable(topi.view(1, len(topi)) )
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			#print(decoder_output)
			if is_eos(topi, batch_size):
				break

	decoder_optimizer.zero_grad()
	encoder_optimizer.zero_grad()
	loss.backward()
	decoder_optimizer.step()
	encoder_optimizer.step()
	
	trglength = len(trg)

	return loss.data[0]/ trglength






# Toy run
train_batch = next(iter(train_iter))
#print("train batch: "+ str(train_batch.get_device()))
src = train_batch.src
trg = train_batch.trg
teacher_forcing_ratio = 1
if torch.cuda.is_available():
	encoder = encoder.cuda()
	decoder = decoder.cuda()
for i in range(0, 1):
	print(toy_train(src, trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion) )




# for i in range(0, 25000):
#  	print(train( encoder, decoder, encoder_optimizer, decoder_optimizer, criterion) )

def train_loop(num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_iter, val_iter):
	losses = []
	total_loss = 0
	for epoch in range(num_epochs):
		new_loss, ec, dc = seq2seq_train(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
		total_loss += new_loss
		losses.append(new_loss)
		if epoch%5==0:
			print("Epoch:"+ str(epoch)+",Average Loss:"+ str(total_loss/5))
			total_loss = 0

