import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import gc
import re
import random

import collections
import numpy as np

from torchtext import data
from torchtext import datasets
from model import EncoderRNN, DecoderRNN
import nltk

FR = data.Field(init_token='<sos>', eos_token='<eos>')
EN = data.Field(init_token='<sos>', eos_token='<eos>')

SOS_token = 2
EOS_token = 3
# batch_size = 5 #switch batch_size to 3 or 5 when using train()

train_set, val_set, test_set = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3

device = 0 if(torch.cuda.is_available()) else -1

def is_eos(topi, batch_size):
	eos_counter = 0
	for i in range(0, batch_size):
		if topi[i][0] == 3 or topi[i][0] == 1 :
			eos_counter += 1

	if eos_counter == batch_size:
		return True
	return False


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

		total_loss += (loss.data[0]/trglength)
		# if b == 300:
		# 	break
		# print(b,' batch complete')
		print(loss.data[0]/trglength)

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


def epoch_training(train_iter, val_iter, num_epoch = 100, learning_rate = 1e-4, hidden_size = 100,  early_stop = False, patience = 10,epsilon = 1e-4):

    # define model
    encoder = EncoderRNN(input_size = len(EN.vocab), hidden_size = hidden_size)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size = len(FR.vocab))

    # define loss criterion
    criterion = nn.NLLLoss()

    # define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    losses = np.ndarray(patience)
    for epoch in range(num_epoch):
        tl = train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print(tl, '**********')
        loss = evaluate(val_iter, encoder, decoder, criterion)
        if early_stop:
            if epoch < patience:
                losses[epoch] = loss
            else:
                if loss+epsilon >= losses.all():
                    print('Stop at Epoch: '+str(epoch)+", With Validation Loss: "+str(loss))
                    break
                else:
                    losses[epoch%patience] = loss

    print('Stop at Epoch: '+str(epoch)+", With Validation Loss: "+str(loss))
    return loss, encoder, decoder


#feed encoder_model, decoder_model, test_set, and device to test
def test_encoder_decoder(encoder, decoder, test_set, device):
	test_iter = data.BucketIterator.splits(test_set, batch_size=1, device = device)
	avg_bleu = 0
	for b, batch in enumerate(test_iter):
		test_batch = batch
		src = test_batch.src
		trg = test_batch.trg

		encoder_hidden = encoder.init_hidden(test_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)

		# decode
		decoder_input = Variable(torch.LongTensor([[SOS_token]*test_iter.batch_size]))

		if torch.cuda.is_available():
			decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*test_iter.batch_size]))

		translated = [SOS_token]
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, context, test_iter.batch_size)

			topv, topi = decoder_output.data.topk(1)
			translated.append(topi[0][0])
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input  = Variable(topi.view(1, len(topi)) )
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			
			if is_eos(topi, test_iter.batch_size):
				break
		english = [EN.vocab.itos[i] for i in src.data[:,0]]
		french_hypothesis = [FR.vocab.itos[i] for i in translated]
		french_reference = [FR.vocab.itos[i] for i in trg.data[:,0]]
		avg_bleu += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis)
	avg_bleu = avg_bleu/len(test_iter)
	return avg_bleu



###################### Main Procedure ##########################################

pars = []
for num_epoch in [100, 500, 1000, 2000]:
    for learning_rate in [1e-4,3e-4,1e-3,3e-3,1e-2]:#,0.05]:
        for hidden_size in [64,128,256]:
            for batch_size in [4,5,10,25,50]:
                for min_freq in [5,50,100,300,500]:
                    pars.append({
                        'num_epoch': num_epoch,
                        'learning_rate': learning_rate,
                        'hidden_size': hidden_size,
                        'batch_size':batch_size,
                        'min_freq':min_freq
                    })

gc.collect()
cnt = 0
base_loss = 10
encoder_model = None
decoder_model = None
optimized_parameters = None
while cnt != 100:
    np.random.seed()
    par = np.random.choice(pars, 1)[0]
    print(str(cnt) +'th trial: \n Parameters: '+ str(par))
    cnt += 1
    EN.build_vocab(train_set.src, min_freq=par['min_freq'])
    FR.build_vocab(train_set.trg, min_freq=par['min_freq'])
    train_iter, val_iter, = data.BucketIterator.splits((train_set, val_set,), batch_sizes=(par['batch_size'], 1,), device = device)
    loss, encoder, decoder = epoch_training(train_iter, val_iter, num_epoch = par['num_epoch'], learning_rate = par['learning_rate'], hidden_size = par['hidden_size'], early_stop = True, patience = 10, epsilon = 1e-4)
    print('\nValidation Loss: '+str(score))
    if loss < base_loss:
        base_loss = loss
        final_par = par
        encoder_model = encoder
        decoder_model = decoder
        optimized_parameters = final_par
    if cnt % 100 == 0:
    	gc.collect()
    cnt += 1
        
    
gc.collect()


print('Optimized Parameters are ', optimized_parameters)
torch.save(encoder_model.state_dict(), 'encoder.pth')
torch.save(decoder_model.state_dict(), 'decoder.pth')

print(test_encoder_decoder(encoder_model, decoder_model, test_set, device))
