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
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
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
eng_word_to_ix, eng_vocab_size = prepare_data.build_dictionary_padding([eng_training_set])
fr_word_to_ix, fr_vocab_size = prepare_data.build_dictionary_padding([fr_training_set])

#Modify training and dev sets by adding tensor representations
prepare_data.sentences_to_padded_index_sequences(eng_word_to_ix, [eng_training_set, eng_dev_set, eng_tst_set ])
prepare_data.sentences_to_padded_index_sequences(fr_word_to_ix, [fr_training_set, fr_dev_set, fr_tst_set ])

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

"""
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
"""

class EncoderRNN(nn.Module):
	def __init__(self, eng_vocab_size, input_size, hidden_size, n_layers):


