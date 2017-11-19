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
        
    def forward(self, input, hidden, batch_size): #hidden should be renamed as context

        output = self.embedding(input)
        hidden = hidden.view(self.n_layers,batch_size,self.hidden_size)
        #for i in range(self.n_layers):
        output = F.relu(output)
        output =  torch.cat((output, hidden), 2)
        output, hidden = self.gru(output, hidden)

        # TODO:
        # output should be once again concatted with context
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

        if torch.cuda.is_available():
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden


class RecurrentMN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, n_gram_size = 4):
        super(RecurrentMN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size) 
        self.lstm = nn.LSTM(hidden_size*2, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size*2, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
        ############## Memory Block ##################################################
        self.m_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.c_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.mb_attn_linear = nn.Linear(n_gram_size, n_gram_size)
        self.mb_gru = nn.GRU(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size, hidden_size)
        ############################################################################## 


    def forward(self, input, hidden, batch_size, n_gram_tensor): #hidden should be renamed as context
        output = self.embedding(input)
        hidden = hidden.view(self.n_layers,batch_size,self.hidden_size)
        output = F.relu(output)
        output =  torch.cat((output, hidden), 2)
        output, (hidden,cell) = self.lstm(output, (hidden, hidden))

        ############## Memory Block ####################################################
        input_memory = self.m_embedding(n_gram_tensor)
        output_memory = self.c_embedding(n_gram_tensor)
        output_memory = output_memory.permute(1,2,0)

        input_memory = input_memory.permute(1,0,2)
        hidden = hidden.permute(1,2,0)

        attn = torch.bmm(input_memory, hidden)# batch_size X n X 1
        attn = attn.squeeze(2)

        lin = self.mb_attn_linear(attn)
        attn_dist = self.softmax(lin)
        attn_dist = attn_dist.view(batch_size, len(n_gram_tensor), -1)

        context = torch.bmm(output_memory, attn_dist).view(-1, batch_size, self.hidden_size)
        hidden = hidden.view(-1, batch_size, self.hidden_size)

        mb_output, mb_hidden = self.mb_gru(context,hidden)
        print(mb_output.size())
        print(output.size())

        #return mb_output
        output = mb_output

        #s should be batch X hidden_size X 1
        ####################################################################################        

        output = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

        if torch.cuda.is_available():
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden




