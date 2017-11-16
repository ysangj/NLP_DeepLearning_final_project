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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(input_size, hidden_size) 
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, source_sentence, hidden):
        seq_length = len(source_sentence)
        embedded = self.embedding(source_sentence)
        output, hidden = self.gru(embedded, hidden) 
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        ## For GPU run only
        if torch.cuda.is_available(): 
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size) if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax().cuda()
        
    def forward(self, input, hidden, batch_size):

        output = self.embedding(input)
        hidden = hidden.view(self.n_layers,batch_size,self.hidden_size)
        #for i in range(self.n_layers):
        output = F.relu(output)
        output =  torch.cat((output, hidden), 2)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        ## For GPU run only
        # if torch.cuda.is_available(): hidden = hidden.cuda()
        if torch.cuda.is_available():
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden
