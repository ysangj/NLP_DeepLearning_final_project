import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import re
import random

import collections
import numpy as np

import re

data_path = 'en-fr/'

def not_contain_unneccessary(example_text):

	if '<url' in example_text:
		return False
	if '<talkid' in example_text:
		return False
	if '<reviewer' in example_text:
		return False
	if '<translator' in example_text:
		return False
	if '<speaker' in example_text:
		return False
	return True


def load_training_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[0:]
            if not_contain_unneccessary(example['text']):
            	example['text'] = example['text'].replace('<title>','')
            	example['text'] = example['text'].replace('</title>','')
            	example['text'] = example['text'].replace('<keywords>','')
            	example['text'] = example['text'].replace('</keywords>','')
            	example['text'] = example['text'].replace('<description>','')
            	example['text'] = example['text'].replace('</description>','')
            	data.append(example)
    return data


PADDING = "<PAD>"
UNKNOWN = "<UNK>"
max_seq_length = 30

def tokenize(string):
    string = string.lower()
    return string.split()


def build_dictionary_padding(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)


def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)



def load_dev_test_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            text = text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text
            if '<seg id' in example['text']:
                example['text'] = re.sub(r'<>', '', example['text'])
                toklist = tokenize(example['text'])
                example['text'] = ' '.join(toklist[2:len(toklist)-1] )
                data.append(example)
    return data

# #Training set
# #Training source sentences, X
# eng_training_set = load_training_data(data_path + '/train.tags.en-fr.en')
# #Training Target Sentence, Y
# fr_training_set = load_training_data(data_path + '/train.tags.en-fr.fr')

# #Development
# #Development source sentences, X
# eng_dev_set = load_dev_test_data(data_path+ '/IWSLT16.TED.dev2010.en-fr.en.XML')
# #Development Target Sentence, Y
# fr_dev_set = load_dev_test_data(data_path+ '/IWSLT16.TED.dev2010.en-fr.fr.XML')

# #Test set
# #Test source sentences, X
# eng_tst_set = load_dev_test_data(data_path+ '/IWSLT16.TED.tst2010.en-fr.en.XML')
# #Test Target Sentence, Y
# fr_tst_set = load_dev_test_data(data_path+ '/IWSLT16.TED.tst2010.en-fr.fr.XML')




# #Create vocabulary(dictionary) using training set
# eng_word_to_ix, eng_vocab_size = build_dictionary_padding([eng_training_set])
# fr_word_to_ix, fr_vocab_size = build_dictionary_padding([fr_training_set])

# #Modify training and dev sets by adding tensor representations
# sentences_to_padded_index_sequences(eng_word_to_ix, [eng_training_set, eng_dev_set, eng_tst_set ])
# sentences_to_padded_index_sequences(fr_word_to_ix, [fr_training_set, fr_dev_set, fr_tst_set ])

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
Each data point in training/dev/test set looks like

{'text': 'thank you very much.', 'text_index_sequence': 

Columns 0 to 10 
 57949   7079  94142  16653      0      0      0      0      0      0      0

Columns 11 to 21 
     0      0      0      0      0      0      0      0      0      0      0

Columns 22 to 29 
     0      0      0      0      0      0      0      0
[torch.LongTensor of size 1x30]
}

Each data set i.e, fr_dev_set is a list of sets


"""

