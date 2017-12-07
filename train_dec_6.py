import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import gc
import re
import random
from six.moves import urllib
import os
import subprocess
import logging
import pickle
import collections
import numpy as np

from torchtext import data
from torchtext import datasets
from model_nov23 import EncoderRNN, DecoderRNN
import nltk

multi_bleu_path, _ = urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
        "master/scripts/generic/multi-bleu.perl")
os.chmod(multi_bleu_path, 0o755)

logging.basicConfig(filename='epoch_5.log',level=logging.WARNING)
logging.warning('Packages Imported')

FR = data.Field(init_token='<sos>', eos_token='<eos>')
EN = data.Field(init_token='<sos>', eos_token='<eos>')

SOS_token = 2
EOS_token = 3

train_set, val_set, test_set = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3

device = 0 if(torch.cuda.is_available()) else -1
logging.warning('Data Loaded')

def is_eos(topi, batch_size):
	eos_counter = 0
	for i in range(0, batch_size):
		if topi[i][0] == 3 or topi[i][0] == 1 :
			eos_counter += 1
	if eos_counter == batch_size:
		return True
	return False


def train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):#, teacher_forcing_ratio=1.0):
	
	decoder_optimizer.zero_grad()
	encoder_optimizer.zero_grad()
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

		decoder_hidden = context

		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, context, train_iter.batch_size)
			topv, topi = decoder_output.data.topk(1)
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input = trg[trg_index].view(1, len(trg[trg_index]))

		loss.backward()
		torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.25)
		torch.nn.utils.clip_grad_norm(decoder.parameters(), 0.25)
		encoder_optimizer.step()
		decoder_optimizer.step()
		trglength = len(trg)

		total_loss += (loss.data[0]/trglength)
		if b %500==0:
			print(b,' batch complete')
			print(loss.data[0]/trglength)
			#break
		if b==len(train_iter)-1:
			break
			
	return total_loss/len(train_iter)


def evaluate(val_iter, encoder, decoder, criterion):
	total_loss = 0
	for b, batch in enumerate(val_iter):
		loss = 0
		val_batch = batch
		src = val_batch.src
		trg = val_batch.trg

		encoder_hidden = encoder.init_hidden(val_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)

		# decode
		decoder_input = Variable(torch.LongTensor([[SOS_token]*val_iter.batch_size]))
		decoder_hidden = context
		if torch.cuda.is_available():
			decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*val_iter.batch_size]))

		translated = []
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, context, val_iter.batch_size)
			topv, topi = decoder_output.data.topk(1)
			translated.append(topi[0][0])
			loss += criterion(decoder_output, trg[trg_index])
			decoder_input  = Variable(topi.view(1, len(topi)) )
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input			
			
			if topi[0][0] == EOS_token:
				break

		trglength = len(trg)
		total_loss += loss.data[0]/trglength
		
		if b% 500 == 0:
			print("[ENGLISH]: ", " ".join([EN.vocab.itos[i] for i in src.data[:,0]]))
			print("[French]: ", " ".join([FR.vocab.itos[i] for i in translated]))
			print("[French Original]: ", " ".join([FR.vocab.itos[i] for i in trg.data[:,0]]))

		if b==len(val_iter)-1:
			print("[ENGLISH]: ", " ".join([EN.vocab.itos[i] for i in src.data[:,0]]))
			print("[French]: ", " ".join([FR.vocab.itos[i] for i in translated]))
			print("[French Original]: ", " ".join([FR.vocab.itos[i] for i in trg.data[:,0]]))
			break
	return total_loss/len(val_iter)


def epoch_training(train_iter, val_iter, num_epoch = 100, learning_rate = 1e-4, hidden_size = 100,  early_stop = False, patience = 2,epsilon = 1e-4):

    # define model
    encoder = EncoderRNN(input_size = len(EN.vocab), hidden_size = hidden_size)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size = len(FR.vocab))

    # define loss criterion
    criterion = nn.NLLLoss()

    # define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    losses = np.ndarray(patience)

    res_loss = 13
    res_encoder = None
    res_decoder = None
    res_epoch = 0

    for epoch in range(num_epoch):
        tl = train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss = evaluate(val_iter, encoder, decoder, criterion)
        logging.warning('************Epoch: ' + str(epoch) + ' Training Loss: '+str(tl)+' Validation Loss: '+str(loss)+'*********')
        #save the model with the lowest validation loss
        if loss < res_loss:
        	res_loss = loss
        	res_encoder = encoder
        	res_decoder = decoder
        	res_epoch = epoch
        	logging.warning('Updated validation loss as ' + str(res_loss) + ' at '+str(res_epoch))

    print('Stop at Epoch: '+str(res_epoch)+", With Validation Loss: "+str(res_loss))
    logging.warning('Stop at Epoch: '+str(res_epoch)+", With Validation Loss: "+str(res_loss))
    return res_loss, res_encoder, res_decoder, loss


#feed encoder_model, decoder_model, test_set, and device to test
def test_encoder_decoder(encoder, decoder, device,test_set):

	test_iter, = data.BucketIterator.splits(
    (test_set,), batch_sizes=(1,), device=device) 

	avg_bleu1 = 0
	avg_bleu2 = 0
	avg_bleu3 = 0
	avg_bleu4 = 0
	avg_bleu5 = 0
	#avg_bleu6 = 0
	avg_bleu7 = 0
	english_test = []
	french_test = []
	french_hypo = []
	for b, batch in enumerate(test_iter):
		test_batch = batch
		src = test_batch.src
		trg = test_batch.trg

		encoder_hidden = encoder.init_hidden(test_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)

		# decode
		decoder_input = Variable(torch.LongTensor([[SOS_token]*test_iter.batch_size]))
		decoder_hidden = context
		if torch.cuda.is_available():
			decoder_input = Variable(torch.cuda.LongTensor([[SOS_token]*test_iter.batch_size]))

		translated = [SOS_token]
		for trg_index in range(1, len(trg)):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, context, test_iter.batch_size)

			topv, topi = decoder_output.data.topk(1)
			translated.append(topi[0][0])
			decoder_input  = Variable(topi.view(1, len(topi)) )
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			
			if is_eos(topi, test_iter.batch_size):
				break
		english = [EN.vocab.itos[i] for i in src.data[:,0]]
		french_hypothesis = [FR.vocab.itos[i] for i in translated]
		french_reference = [FR.vocab.itos[i] for i in trg.data[:,0]]
		english_test.append(english)
		french_test.append(french_reference)
		french_hypo.append(french_hypothesis)
		avg_bleu1 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
		avg_bleu2 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)
		avg_bleu3 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method3)
		avg_bleu4 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
		avg_bleu5 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method5)
		#avg_bleu6 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method6)
		avg_bleu7 += nltk.translate.bleu_score.sentence_bleu([french_reference], french_hypothesis,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7)        
		if b==len(test_iter)-1:
			break 
	logging.warning(avg_bleu1/len(test_iter))
	logging.warning(avg_bleu2/len(test_iter))
	logging.warning(avg_bleu3/len(test_iter))
	logging.warning(avg_bleu4/len(test_iter))
	logging.warning(avg_bleu5/len(test_iter))
	#logging.warning(avg_bleu6/len(test_iter))
	logging.warning(avg_bleu7/len(test_iter))
	return french_test, french_hypo


###################### Main Procedure ##########################################

pars = []
for num_epoch in [5]:
    for learning_rate in [3e-4]:#,0.05]:
        for hidden_size in [128]:
            for batch_size in [10]:
                for min_freq in [250]:
                    pars.append({
                        'num_epoch': num_epoch,
                        'learning_rate': learning_rate,
                        'hidden_size': hidden_size,
                        'batch_size':batch_size,
                        'min_freq':min_freq
                    })



gc.collect()
cnt = 0
base_loss = 30
encoder_model = None
decoder_model = None
optimized_parameters = None

while cnt < 1:
    np.random.seed()
    par = np.random.choice(pars, 1)[0]
    logging.warning(str(cnt) +'th trial: \n Parameters: '+ str(par))
    cnt += 1
    EN.build_vocab(train_set.src, min_freq=par['min_freq'])
    FR.build_vocab(train_set.trg, min_freq=par['min_freq'])
    train_iter, val_iter = data.BucketIterator.splits((train_set, val_set), sort_key = lambda ex: data.interleave_keys(len(ex.src), len(ex.trg)), batch_sizes=(par['batch_size'], 1), device = device)
    loss, encoder, decoder = epoch_training(train_iter, val_iter, num_epoch = par['num_epoch'], learning_rate = par['learning_rate'], hidden_size = par['hidden_size'], early_stop = False, patience = 2, epsilon = 1e-4)
    logging.warning('\nValidation Loss: '+str(loss))

    if loss < base_loss:
        base_loss = loss
        final_par = par
        encoder_model = encoder
        decoder_model = decoder
        optimized_parameters = final_par
    if cnt % 1 == 0:
        gc.collect()
        
logging.warning('Optimized Parameters are '+str(optimized_parameters))


pickle.dump(encoder_model,open('encoder_epoch_5.p','wb'))
pickle.dump(decoder_model,open('decoder_epoch_5.p','wb'))

french_test, french_hypo = test_encoder_decoder(encoder_model, decoder_model, device,test_set)

pickle.dump(french_test,open('french_test_epoch_5.p','wb'))
pickle.dump(french_hypo,open('french_hypo_epoch_5.p','wb'))


sentence1 = ''
for j in range(len(french_test)):
    sentence1 += ' '.join(french_test[j][i] for i in range(len(french_test[j])))
    sentence1 = sentence1+'\n'
text_file = open("french_test_epoch_5.txt", "w")
text_file.write(sentence1)
text_file.close()

sentence2 = ''
for j in range(len(french_hypo)):
    sentence2 += ' '.join(french_hypo[j][i] for i in range(len(french_hypo[j])))
    sentence2 = sentence2+'\n'
text_file = open("french_hypo_epoch_5.txt", "w")
text_file.write(sentence2)
text_file.close()


lowercase = 1
with open('french_hypo_epoch_5.txt', "r") as read_pred:
    bleu_cmd = [multi_bleu_path]
    if lowercase:
        bleu_cmd += ["-lc"]
    bleu_cmd += ['french_test_epoch_5.txt']
    try:
        bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        bleu_score = float(bleu_score)
        logging.warning(str(bleu_score))
        logging.warning(bleu_out)
    except subprocess.CalledProcessError as error:
        if error.output is not None:
            logging.warning("multi-bleu.perl script returned non-zero exit code")
            logging.warning(error.output)
        bleu_score = np.float32(0.0)

