import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import re
import random

import collections
import numpy as np

import prepare_data


data_path = 'en-fr/'
# PADDING = "<PAD>"
# UNKNOWN = "<UNK>"
max_seq_length = 30

eng_training_set = prepare_data.load_training_data(data_path + '/train.tags.en-fr.en')
#Training Target Sentence, Y
fr_training_set = prepare_data.load_training_data(data_path + '/train.tags.en-fr.fr')

#Development
#Development source sentences, X
eng_dev_set = prepare_data.load_dev_test_data(data_path+ '/IWSLT16.TED.dev2010.en-fr.en.XML')
#Development Target Sentence, Y
fr_dev_set = prepare_data.load_dev_test_data(data_path+ '/IWSLT16.TED.dev2010.en-fr.fr.XML')

#Test set
#Test source sentences, X
eng_tst_set = prepare_data.load_dev_test_data(data_path+ '/IWSLT16.TED.tst2010.en-fr.en.XML')
#Test Target Sentence, Y
fr_tst_set = prepare_data.load_dev_test_data(data_path+ '/IWSLT16.TED.tst2010.en-fr.fr.XML')


#Create vocabulary(dictionary) using training set
eng_word_to_ix, eng_vocab_size, eng_vocabulary = prepare_data.build_dictionary_padding([eng_training_set])
fr_word_to_ix, fr_vocab_size, fr_vocabulary = prepare_data.build_dictionary_padding([fr_training_set])

#Modify training and dev sets by adding tensor representations
prepare_data.sentences_to_padded_index_sequences(eng_word_to_ix, [eng_training_set, eng_dev_set, eng_tst_set ])
prepare_data.sentences_to_padded_index_sequences(fr_word_to_ix, [fr_training_set, fr_dev_set, fr_tst_set ])



def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield [source[index] for index in batch_indices]


def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
    return batches


def get_batch(batch):
    vectors = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
    return vectors, labels
# print(eng_training_set)
# print("11111111111111111111111111111111111111111111111")
# print(fr_training_set)
# print("2222222222222222222222222222222222222222222222222")
# print(eng_dev_set)
# print("3333333333333333333333333333333333333333333333333")
# print(fr_dev_set)
# print("444444444444444444444444444444444444444444444444444")
# print(eng_tst_set)
# print("5555555555555555555555555555555555555555555555555")
# print(fr_tst_set)
# print("666666666666666666666666666666666666666666666666")

##Pleas Use Variable
# our_tensor = Variable( eng_training_set[5]['text_index_sequence'] )  
# emb = nn.Embedding(eng_vocab_size, 100)

# print(emb(our_tensor).view(30, 1, -1))

class EncoderRNN(nn.Module):
	# In our model, we set hidden dimension as same as the embedding dimension
    def __init__(self, vocab_size, embedding_dim, n_layers=1000):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.gru = nn.GRU(embedding_dim, embedding_dim, n_layers)
        
    def forward(self, source_sentence, hidden):
        # Note: we run this all at once (over the whole input sequence)

        #seq_len = max_seq_length # source_sentence should be a 30 by 1 tensor

        embedded = self.embedding(word_inputs).view(max_seq_length, 1, -1) # Embed each element in source_sentence.

        output, hidden = self.gru(embedded, hidden) 
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, 1, self.embedding_dim))
        ## For GPU run only
        # if USE_CUDA: hidden = hidden.cuda()
        # return hidden





















# class ElmanRNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, batch_size):
#         super(ElmanRNN, self).__init__()
        
#         self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.embedding_size = embedding_dim
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.batch_size = batch_size
        
#         self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
#         self.decoder = nn.Linear(hidden_size, output_size)
#         self.init_weights()
    
#     def forward(self, x, hidden):
#         x_emb = self.embed(x)                
#         embs = torch.chunk(x_emb, x_emb.size()[1], 1)
        
#         def step(emb, hid):
#             combined = torch.cat((hid, emb), 1)
#             hid = F.tanh(self.i2h(combined))
#             return hid

#         for i in range(len(embs)):
#             hidden = step(embs[i].squeeze(), hidden)
        
#         output = self.decoder(hidden)
#         return output, hidden

#     def init_hidden(self):
#         h0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
#         return h0
    
#     def init_weights(self):
#         initrange = 0.1
#         lin_layers = [self.i2h, self.decoder]
#         em_layer = [self.embed]
     
#         for layer in lin_layers+em_layer:
#             layer.weight.data.uniform_(-initrange, initrange)
#             if layer in lin_layers:
#                 layer.bias.data.fill_(0)
