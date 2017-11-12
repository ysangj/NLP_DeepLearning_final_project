import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import gc
import re
import random
import pickle
import collections
import numpy as np

from torchtext import data
from torchtext import datasets
from model import EncoderRNN, DecoderRNN
from train import early_stop_patience, evaluate, train
import queue

FR = data.Field(init_token='<sos>', eos_token='<eos>')
EN = data.Field(init_token='<sos>', eos_token='<eos>')

SOS_token = 2
EOS_token = 3
# batch_size = 5 #switch batch_size to 3 or 5 when using train()

training, val, test = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3


#hp
learning_rate=0.0001
EN.build_vocab(training.src, min_freq=50)
FR.build_vocab(training.trg, min_freq=50)
hidden_size = 100

device = 0 if(torch.cuda.is_available()) else -1

train_iter, val_iter, test_iter = data.BucketIterator.splits(
(training, val, test), batch_sizes=(10, 6, 5), device=device) #set device as any number other than -1 in cuda envicornment



def epoch_training(num_epoch = 100, learning_rate = 1e-4, hidden_size = 100,  early_stop = False, patience = 10,epsilon = 1e-4, train_iter = train_iter, val_iter = val_iter):

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
        train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss = evaluate(val_iter, encoder, decoder, criterion)
        if early_stop:
            if epoch < patience:
                losses[epoch] = loss
            else:
                if loss+episilon >= losses.all():
                    print('Stop at Epoch: '+str(epoch)+", With Validation Loss: "+str(loss))
                    break
                    
#                     pickle.dump(encoder,file = open('encoder.p','wb'))
#                     pickle.dump(decoder,file = open('decoder.p','wb'))
#                     return looses.min()
                else:
                    losses[epoch%patience] = loss
#     pickle.dump(encoder,file = open('encoder.p','wb'))
#     pickle.dump(decoder,file = open('decoder.p','wb'))
    print('Stop at Epoch: '+str(epoch)+", With Validation Loss: "+str(loss))
    return loss
        

pars = []
for num_epoch in [100, 500, 1000, 2000]:
    for learning_rate in [1e-4,1e-3,1e-2,0.05,0.1,1]:
        for hidden_size in [64,128,256,512]:
            for batch_size in [3,6,10,20]:
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
base_score = 10
while cnt != 1:
    np.random.seed()
    par = np.random.choice(pars, 1)[0]
    print(str(cnt) +'th trial: \n Parameters: '+ str(par))
    cnt += 1
    EN.build_vocab(training.src, min_freq=par['min_freq'])
    FR.build_vocab(training.trg, min_freq=par['min_freq'])
    train_iter, val_iter = data.BucketIterator.splits(
    (training, val), batch_sizes=(par['batch_size'],par['batch_size']), device=device) #set device as any number other than -1 in cuda envicornment
    score = epoch_training(num_epoch = par['num_epoch'], learning_rate = par['learning_rate'], hidden_size = par['hidden_size'],  early_stop = True, patience = 10,epsilon = 1e-4, train_iter = train_iter, val_iter = val_iter)
    print('\nValidation Loss: '+str(score))
    if score < base_score:
        base_score = score
        final_par = par        
    if cnt%100 == 0:
        gc.collect()
    print('Final Parameter: '+str(final_par))
gc.collect()    
