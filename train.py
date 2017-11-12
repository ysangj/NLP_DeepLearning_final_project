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
import Queue

FR = data.Field(init_token='<sos>', eos_token='<eos>')
EN = data.Field(init_token='<sos>', eos_token='<eos>')

SOS_token = 2
EOS_token = 3
# batch_size = 5 #switch batch_size to 3 or 5 when using train()

train, val, test = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3


#hp
learning_rate=0.0001
EN.build_vocab(train.src, min_freq=50)
FR.build_vocab(train.trg, min_freq=50)
hidden_size = 100

device = 0 if(torch.cuda.is_available()) else -1

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(10, 6, 5), device=device) #set device as any number other than -1 in cuda envicornment

# define model
encoder = EncoderRNN(input_size = len(EN.vocab), hidden_size = hidden_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size = len(FR.vocab))

# define loss criterion
criterion = nn.NLLLoss()

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


def toy_train(src, trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch, teacher_forcing_ratio=1.0 ):
	loss = 0
	# encode
	encoder_hidden = encoder.init_hidden(batch_size)
	encoder_out, context = encoder(train_batch.src, encoder_hidden)
	#context = contexts[len(contexts)-1]

	# decoder
	decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]))

	if torch.cuda.is_available():
		decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*batch_size]))

	translated = []
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	# if use_teacher_forcing:
	for trg_index in range(1, len(trg)):
		decoder_output, decoder_hidden = decoder(decoder_input, context, batch_size)
		topv, topi = decoder_output.data.topk(1)
		translated.append(topi[0][0])
		loss += criterion(decoder_output, trg[trg_index])
		decoder_input = trg[trg_index].view(1, len(trg[trg_index]))

	decoder_optimizer.zero_grad()
	encoder_optimizer.zero_grad()

	loss.backward()
	decoder_optimizer.step()
	encoder_optimizer.step()
	trglength = len(trg)
	if epoch % 100 == 0:
		print("[ENGLISH]: ", " ".join([EN.vocab.itos[i] for i in src.data[:,0]]))
		print("[French]: ", " ".join([FR.vocab.itos[i] for i in translated]))
		print("[French Original]: ", " ".join([FR.vocab.itos[i] for i in trg.data[:,0]]))
	return loss.data[0]/ trglength


def train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):#, teacher_forcing_ratio=1.0):

	total_loss = 0
	for b, batch in enumerate(train_iter):
		loss = 0
		train_batch = batch
		src = train_batch.src
		trg = train_batch.trg

		# encode
		encoder_hidden = encoder.init_hidden(train_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)

		# decode
		decoder_input = Variable(torch.LongTensor([[SOS_token]*train_iter.batch_size]))

		if torch.cuda.is_available():
			decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*train_iter.batch_size]))

		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, train_iter.batch_size)
			topv, topi = decoder_output.data.topk(1)
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input = trg[trg_index].view(1, len(trg[trg_index]))

		decoder_optimizer.zero_grad()
		encoder_optimizer.zero_grad()
		loss.backward()
		decoder_optimizer.step()
		encoder_optimizer.step()
		trglength = len(trg)

		total_loss += loss.data[0]
		if b == 300:
			break
		print(b,' batch complete')

	return total_loss/len(train_iter)


def evaluate(val_iter, encoder, decoder, criterion):
	total_loss = 0
	for b, batch in enumerate(val_iter):
		loss = 0
		val_batch = batch
		src = val_batch.src
		trg = val_batch.trg
				# encode
		encoder_hidden = encoder.init_hidden(val_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)

		# decode
		decoder_input = Variable(torch.LongTensor([[SOS_token]*val_iter.batch_size]))

		if torch.cuda.is_available():
			decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*val_iter.batch_size]))

		translated = []
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, val_iter.batch_size)

			topv, topi = decoder_output.data.topk(1)
			translated.append(topi[0][0])
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input  = Variable(topi.view(1, len(topi)) )
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			
			if is_eos(topi, val_iter.batch_size):
				break

		trglength = len(trg)

		if b % 200 == 0:
			print("[ENGLISH]: ", " ".join([EN.vocab.itos[i] for i in src.data[:,0]]))
			print("[French]: ", " ".join([FR.vocab.itos[i] for i in translated]))
			print("[French Original]: ", " ".join([FR.vocab.itos[i] for i in trg.data[:,0]]))
		total_loss += loss.data[0]

	return total_loss/len(train_iter)


def early_stop_patience(val_loss_q, patience=10):
	minimum = -1
	for loss in IterableQueue(val_loss_q):
		if loss> minimum:
			minimum = loss
		else:
			return False
	return True


	
print(evaluate(val_iter, encoder, decoder, criterion))
print(train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion))
print(evaluate(val_iter, encoder, decoder, criterion))
