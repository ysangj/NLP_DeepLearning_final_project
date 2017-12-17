import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import gc
import re
import random
import logging
import pickle
import collections
import numpy as np

from torchtext import data
from torchtext import datasets
from model import EncoderRNN, RecurrentMemory
import queue
import nltk
from collections import deque

from six.moves import urllib
import os
import subprocess
import logging
import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

multi_bleu_path, _ = urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
        "master/scripts/generic/multi-bleu.perl")
os.chmod(multi_bleu_path, 0o755)

file_name = 'dec15_rmn_iwslt_3_en_de'
logging.basicConfig(filename=file_name+'.log',level=logging.WARNING)
logging.warning('Packages Imported')

DE = data.Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', fix_length = 13)
EN = data.Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', fix_length = 13)

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3
train_set, val_set, test_set = datasets.IWSLT.splits(exts=('.en', '.de'), fields=(EN, DE))

device = 0 if(torch.cuda.is_available()) else -1
logging.warning('Data Loaded')
logging.warning('Embedding Size = 128, Fix Length = 13')
def queue2tensor(memory_queue):
	tensor = memory_queue[0]
	for i in range(1,len(memory_queue)):
		tensor = torch.cat((tensor, memory_queue[i]), 0)
	return tensor

def is_eos(topi, batch_size):
	eos_counter = 0
	for i in range(0, batch_size):
		if topi[i][0] == 3 or topi[i][0] == 1 :
			eos_counter += 1

	if eos_counter == batch_size:
		return True
	return False

def compute_bleu(hypothesis_file_name, test_filename_name):
	lowercase = 1
	bleu_score = 0
	with open(hypothesis_file_name, "r") as read_pred:
	    bleu_cmd = [multi_bleu_path]
	    if lowercase:
	        bleu_cmd += ["-lc"]
	    bleu_cmd += [test_filename_name]
	    try:
	        bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
	        bleu_out = bleu_out.decode("utf-8")
	        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
	        bleu_score = float(bleu_score)

	    except subprocess.CalledProcessError as error:
	        if error.output is not None:
	            logging.warning("multi-bleu.perl script returned non-zero exit code")
	            logging.warning(error.output)
	        bleu_score = np.float32(0.0)
	return bleu_score,bleu_out

def rm_train(train_iter, encoder, recurrent_memory, encoder_optimizer, recurrent_memory_optimizer, criterion, hidden_size, memory_size):
	
	total_loss = 0
	for b, batch in enumerate(train_iter):
		loss = 0
		train_batch = batch
		src = train_batch.src
		trg = train_batch.trg

		encoder_hidden = encoder.init_hidden(train_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)
		context = context[-1].unsqueeze(0)

		# decoder
		recurrent_memory_input = Variable(torch.LongTensor([[SOS_token]*train_iter.batch_size]))
		if torch.cuda.is_available():
			recurrent_memory_input = Variable(torch.cuda.LongTensor([[SOS_token]*train_iter.batch_size]))
		recurrent_memory_hidden = context

		memory_queue = deque(maxlen = memory_size)
		for i in range(0,memory_size):
			if torch.cuda.is_available():
				memory_queue.append(Variable(torch.cuda.LongTensor([[PAD_token]*train_iter.batch_size])))
			else:
				memory_queue.append(Variable(torch.LongTensor([[PAD_token]*train_iter.batch_size])))
		memory_queue.append(recurrent_memory_input)
		
		cell = Variable(torch.zeros(1, train_iter.batch_size, hidden_size)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(1, train_iter.batch_size, hidden_size))

		# Just use Teacher Forcing
		for trg_index in range(1, len(trg)):
			memory_tensor = queue2tensor(memory_queue)
			recurrent_memory_output, recurrent_memory_hidden, cell = recurrent_memory(recurrent_memory_input, recurrent_memory_hidden, cell, context, train_iter.batch_size, memory_tensor, memory_size)
			topv, topi = recurrent_memory_output.data.topk(1)
			loss += criterion(recurrent_memory_output, trg[trg_index])
			recurrent_memory_input = trg[trg_index].view(1, len(trg[trg_index]))
			memory_queue.append(trg[trg_index].view(1, len(trg[trg_index])))

		recurrent_memory_optimizer.zero_grad()
		encoder_optimizer.zero_grad()
		loss.backward()

		#Gradient Clipping
		torch.nn.utils.clip_grad_norm(encoder.parameters(), 1.0)
		torch.nn.utils.clip_grad_norm(recurrent_memory.parameters(), 1.0)

		recurrent_memory_optimizer.step()
		encoder_optimizer.step()
		trglength = len(trg)

		total_loss += loss.data[0]/ trglength
		if b % 500 == 0:
			logging.warning(str(b)+' batch complete')
			logging.warning(loss.data[0]/trglength)
			#break
		if b==len(train_iter)-1:
			break
	return total_loss/len(train_iter)

def evaluate(val_iter, encoder, recurrent_memory, hidden_size, memory_size, criterion):
	total_loss = 0
	de_test = []
	de_hypo = []
	for b, batch in enumerate(val_iter):
		loss = 0
		val_batch = batch
		src = val_batch.src
		trg = val_batch.trg
		
		# encode
		encoder_hidden = encoder.init_hidden(val_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)
		context = context[-1].unsqueeze(0)

		# decode
		recurrent_memory_input = Variable(torch.LongTensor([[SOS_token]*val_iter.batch_size]))
		recurrent_memory_hidden = context
		if torch.cuda.is_available():
			recurrent_memory_input = Variable(torch.cuda.LongTensor([[SOS_token]*val_iter.batch_size]))

		memory_queue = deque(maxlen = memory_size)
		for i in range(0,memory_size):
			if torch.cuda.is_available():
				memory_queue.append(Variable(torch.cuda.LongTensor([[PAD_token]*val_iter.batch_size])))
			else:
				memory_queue.append(Variable(torch.LongTensor([[PAD_token]*val_iter.batch_size])))
		memory_queue.append(recurrent_memory_input)

		translated = [SOS_token]
		cell = Variable(torch.zeros(1, val_iter.batch_size, hidden_size)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(1, val_iter.batch_size, hidden_size))

		for trg_index in range(1, len(trg)):
			
			memory_tensor = queue2tensor(memory_queue)
			recurrent_memory_output, recurrent_memory_hidden, cell = recurrent_memory(recurrent_memory_input, recurrent_memory_hidden, cell, context, val_iter.batch_size, memory_tensor, memory_size)
			
			topv, topi = recurrent_memory_output.data.topk(1)
			translated.append(topi[0][0])
			loss += criterion(recurrent_memory_output, trg[trg_index])

			recurrent_memory_input = Variable(topi.view(1, len(topi)) )
			memory_queue.append(trg[trg_index].view(1, len(trg[trg_index])))
			
			if topi[0][0] == EOS_token:
				break

		trglength = len(trg)
		total_loss += loss.data[0]/trglength

		hyp = [DE.vocab.itos[i] for i in translated]
		ref = [DE.vocab.itos[i] for i in trg.data[:,0]]
		de_test.append(ref)
		de_hypo.append(hyp)

		if b% 500 == 0:
			print("[ENGLISH]: ", " ".join([EN.vocab.itos[i] for i in src.data[:,0]]))
			print("[DE]: ", " ".join([DE.vocab.itos[i] for i in translated]))
			print("[DE Original]: ", " ".join([DE.vocab.itos[i] for i in trg.data[:,0]]))
		if b==len(val_iter)-1:
			break

	sentence1 = ''
	for j in range(len(de_test)):
	    sentence1 += ' '.join(de_test[j][i] for i in range(len(de_test[j])))
	    sentence1 = sentence1+'\n'
	text_file = open("de_val_test_"+file_name+".txt", "w")
	text_file.write(sentence1)
	text_file.close()

	sentence2 = ''
	for j in range(len(de_hypo)):
	    sentence2 += ' '.join(de_hypo[j][i] for i in range(len(de_hypo[j])))
	    sentence2 = sentence2+'\n'
	text_file = open("de_val_hypo_"+file_name+".txt", "w")
	text_file.write(sentence2)
	text_file.close()

	val_bleu = compute_bleu("de_val_hypo_"+file_name+".txt", "de_val_test_"+file_name+".txt")[0]
	return total_loss/len(val_iter), val_bleu

def epoch_training(train_iter, val_iter, num_epoch = 100, learning_rate = 1e-4, hidden_size = 1000,  early_stop = False, patience = 4, epsilon = 1e-4, memory_size = 7):

    # define model
    encoder = EncoderRNN(input_size = len(EN.vocab), hidden_size = hidden_size)
    recurrent_memory = RecurrentMemory(hidden_size=hidden_size, output_size = len(DE.vocab), memory_size = memory_size)

    # define loss criterion
    criterion = nn.NLLLoss( ignore_index = PAD_token)

    # define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    recurrent_memory_optimizer = optim.Adam(recurrent_memory.parameters(), lr=learning_rate)
    losses = np.ndarray(patience)

    res_loss = 13
    res_encoder = None
    res_rm = None
    res_epoch = 0
    base_bleu = 0

    for epoch in range(num_epoch):
        tl = rm_train(train_iter, encoder, recurrent_memory, encoder_optimizer, recurrent_memory_optimizer, criterion, hidden_size, memory_size )
        loss, val_bleu = evaluate(val_iter, encoder, recurrent_memory, hidden_size, memory_size, criterion)
        logging.warning('************Epoch: '+str(epoch)+' Training Loss: '+str(tl)+' Validation Loss: '+str(loss)+' Validation Bleu: '+str(val_bleu)+'*********')
        
        if base_bleu <= val_bleu:
        	base_bleu = val_bleu
        	res_loss = loss
        	res_encoder = encoder
        	res_rm = recurrent_memory
        	res_epoch = epoch
        	not_updated = 0
        	logging.warning('Updated validation loss as ' + str(res_loss) + ' at '+str(res_epoch)+' with validation bleu: '+str(base_bleu))
        else:
        	not_updated += 1
        if not_updated == patience:
			break
    print('Stop at Epoch: '+str(res_epoch)+", With Validation Loss: "+str(res_loss)+' Validation Bleu: '+str(base_bleu))
    logging.warning('Stop at Epoch: '+str(epoch)+", With Validation Loss: "+str(loss)+' Validation Bleu: '+str(base_bleu))
    return res_loss, res_encoder, res_rm, base_bleu

def test(encoder, recurrent_memory, device, test_set, hidden_size, memory_size):
	test_iter, = data.BucketIterator.splits((test_set,), batch_sizes=(1,), device=device)
	avg_bleu1 = 0
	avg_bleu2 = 0
	avg_bleu3 = 0
	avg_bleu4 = 0
	avg_bleu5 = 0
	avg_bleu7 = 0
	english_test = []
	de_test = []
	de_hypo = []
	for b, batch in enumerate(test_iter):
		test_batch = batch
		src = test_batch.src
		trg = test_batch.trg

		encoder_hidden = encoder.init_hidden(test_iter.batch_size)
		encoder_out, context = encoder(src, encoder_hidden)
		context = context[-1].unsqueeze(0)

		recurrent_memory_input = Variable(torch.cuda.LongTensor([[SOS_token]*test_iter.batch_size])) if torch.cuda.is_available() else Variable(torch.LongTensor([[SOS_token]*test_iter.batch_size]))
		recurrent_memory_hidden = context

		memory_queue = deque(maxlen = memory_size)
		for i in range(0,memory_size):
			if torch.cuda.is_available():
				memory_queue.append(Variable(torch.cuda.LongTensor([[PAD_token]*test_iter.batch_size])))
			else:
				memory_queue.append(Variable(torch.LongTensor([[PAD_token]*test_iter.batch_size])))
		
		memory_queue.append(recurrent_memory_input)
		translated = [SOS_token]
		cell = Variable(torch.zeros(1, test_iter.batch_size, hidden_size)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(1, test_iter.batch_size, hidden_size))

		for trg_index in range(1, len(trg)):
			memory_tensor = queue2tensor(memory_queue)
			recurrent_memory_output, recurrent_memory_hidden, cell = recurrent_memory(recurrent_memory_input, recurrent_memory_hidden, cell, context, test_iter.batch_size, memory_tensor, memory_size)
			topv, topi = recurrent_memory_output.data.topk(1)
			translated.append(topi[0][0])
			recurrent_memory_input = Variable(topi.view(1, len(topi)) )
			memory_queue.append(trg[trg_index].view(1, len(trg[trg_index])))
			
			if topi[0][0] == EOS_token:
				break

		english = [EN.vocab.itos[i] for i in src.data[:,0]]
		de_hypothesis = [DE.vocab.itos[i] for i in translated]
		de_reference = [DE.vocab.itos[i] for i in trg.data[:,0]]
		english_test.append(english)
		de_test.append(de_reference)
		de_hypo.append(de_hypothesis)

		avg_bleu1 += nltk.translate.bleu_score.sentence_bleu([de_reference], de_hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
		avg_bleu2 += nltk.translate.bleu_score.sentence_bleu([de_reference], de_hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)
		avg_bleu3 += nltk.translate.bleu_score.sentence_bleu([de_reference], de_hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method3)
		avg_bleu4 += nltk.translate.bleu_score.sentence_bleu([de_reference], de_hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
		avg_bleu5 += nltk.translate.bleu_score.sentence_bleu([de_reference], de_hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method5)
		avg_bleu7 += nltk.translate.bleu_score.sentence_bleu([de_reference], de_hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7)        

		if b==len(test_iter)-1:
			break

	avg_bleu1 = avg_bleu1/len(test_iter)
	avg_bleu2 = avg_bleu2/len(test_iter)
	avg_bleu3 = avg_bleu3/len(test_iter)
	avg_bleu4 = avg_bleu4/len(test_iter)
	avg_bleu5 = avg_bleu5/len(test_iter)
	avg_bleu7 = avg_bleu7/len(test_iter)

	logging.warning(avg_bleu1)
	logging.warning(avg_bleu2)
	logging.warning(avg_bleu3)
	logging.warning(avg_bleu4)
	logging.warning(avg_bleu5)
	logging.warning(avg_bleu7)

	return de_test, de_hypo, english_test, [avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_bleu5, avg_bleu7]

############################################################### MAIN PROCEDURE ########################################################################## 


############################################################### Train/Validate Model and do Model Selection via Random Search ###################################
pars = []
for num_epoch in [50]:
    for learning_rate in [1e-4]:#,0.05]:
        for hidden_size in [1000]:
            for batch_size in [12]:
                for voc_size in [58000]:
                        for memory_size in [7]:
                            pars.append({
                                'num_epoch': num_epoch,
                                'learning_rate': learning_rate,
                                'hidden_size': hidden_size,
                                'batch_size':batch_size,
                                'voc_size':voc_size,
                                'memory_size':memory_size
                            })
gc.collect()
cnt = 0
base_loss = 100
encoder_model = None
recurrent_memory_model = None
optimized_parameters = None

while cnt <1 :
	cnt += 1
	np.random.seed()
	par = np.random.choice(pars, 1)[0]
	logging.warning(str(cnt) +'th trial: \n Parameters: '+ str(par))
	EN.build_vocab(train_set.src, min_freq=2)
	DE.build_vocab(train_set.trg, max_size=par['voc_size'])

	train_iter, val_iter = data.BucketIterator.splits((train_set, val_set), sort_key = lambda ex: data.interleave_keys(len(ex.src), len(ex.trg)), batch_sizes=(par['batch_size'], 1), device = device)
	loss, encoder, recurrent_memory,bleu = epoch_training(train_iter, val_iter, num_epoch = par['num_epoch'], learning_rate = par['learning_rate'], hidden_size = par['hidden_size'], early_stop = False, patience = 5, epsilon = 1e-4, memory_size = par['memory_size'])
	logging.warning('\nValidation Loss: '+str(loss)+' Validation Bleu: '+str(bleu))

	if loss < base_loss:
		base_loss = loss
		encoder_model = encoder
		recurrent_memory_model = recurrent_memory
		optimized_parameters = par
	if cnt % 1 == 0:
		gc.collect()

gc.collect()
logging.warning('Optimized Parameters are ' + str(optimized_parameters))

############################################### Test Model by computing BLEU score ###########################
de_reference, de_hypo, english_test, nltk_bleus = test(encoder_model, recurrent_memory_model, device, test_set, optimized_parameters['hidden_size'], optimized_parameters['memory_size'] )

print('NLTK Bleu Scores with different smoothing methods are as following. ', nltk_bleus)

english = ''
for j in range(len(english_test)):
    english += ' '.join(english_test[j][i] for i in range(len(english_test[j])))
    english = english + '\n'
text_file = open("english_test_"+file_name+".txt", "w")
text_file.write(english)
text_file.close()

reference = ''
for j in range(len(de_reference)):
    reference += ' '.join(de_reference[j][i] for i in range(len(de_reference[j])))
    reference = reference + '\n'
text_file = open("de_reference_"+file_name+".txt", "w")
text_file.write(reference)
text_file.close()

hypothesis = ''
for j in range(len(de_hypo)):
    hypothesis += ' '.join(de_hypo[j][i] for i in range(len(de_hypo[j])))
    hypothesis = hypothesis + '\n'
text_file = open("de_hypo_"+file_name+".txt", "w")
text_file.write(hypothesis)
text_file.close()

bleu_score,bleu_out = compute_bleu("de_hypo_"+file_name+".txt", "de_reference_"+file_name+".txt")

logging.warning(str(bleu_score))
logging.warning(bleu_out)

pickle.dump(encoder_model,open('/scratch/sjy269/encoder_'+file_name+'.p','wb'))
pickle.dump(recurrent_memory_model,open('/scratch/sjy269/recurrent_memory_'+file_name+'.p','wb'))
