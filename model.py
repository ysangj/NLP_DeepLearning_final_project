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
        #self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size, hidden_size, n_layers)

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
        
    def forward(self, input, hidden, context, batch_size): #hidden should be renamed as context

        output = self.embedding(input)
        hidden = hidden.view(self.n_layers,batch_size,self.hidden_size)
        #for i in range(self.n_layers):
        output = F.relu(output)
        output =  torch.cat((output, context), 2)
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








#V3
class RecurrentMN(nn.Module):
    def __init__(self, hidden_size, output_size,  memory_size, n_layers=1):
        super(RecurrentMN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size) 
        self.lstm = nn.LSTM(hidden_size*2, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size*2, hidden_size, n_layers)
        self.gru = nn.GRU(hidden_size*2, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size*2, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
        ############## Memory Block ##################################################
        self.m_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.c_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)

        self.mb_attn_linear = nn.Linear(1, memory_size)

        self.mb_gru = nn.GRU(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size, hidden_size)
        self.second_lstm = nn.LSTM(hidden_size, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size, hidden_size, n_layers)
        ############################################################################## 


    def forward(self, input, rmn_hidden, cell, context, batch_size, memory_tensor, memory_size): #hidden should be renamed as context
        output = self.embedding(input)
        context = context.view(self.n_layers,batch_size,self.hidden_size)
        output = F.relu(output)
        output =  torch.cat((output, context), 2)


        lstm_output, (lstm_hidden,cell) = self.lstm(output, (rmn_hidden, cell))
        # lstm_output, lstm_hidden = self.gru(output, rmn_hidden)
        ############## Memory Block ####################################################

        input_memory = self.m_embedding(memory_tensor) #M 
        output_memory = self.c_embedding(memory_tensor) #C

        attn = torch.bmm(input_memory.permute(1,0,2), lstm_hidden.permute(1,2,0))# batch_size X n X 1#lstm_hidden.permute(1,2,0)
        attn = attn.squeeze(2) #context?

        # self.mb_attn_linear = nn.Linear(len(memory_tensor), len(memory_tensor))
        # attn = self.mb_attn_linear(attn)
        attn_dist = self.softmax(attn).unsqueeze(2)
        output_memory = output_memory.permute(1,2,0)

        memory_context = torch.bmm(output_memory,  attn_dist).permute(2,0,1)

        mb_hidden, mb_output = self.mb_gru(memory_context, lstm_hidden)
        # mb_output, mb_hidden = self.mb_gru(memory_context, lstm_hidden)
        output = mb_output
        ####################################################################################
        
        output = torch.cat((output, context),2)
        output = self.softmax(self.out(output[0]))
        # output = self.softmax(self.out(output[0]))


        # return output, hidden
        return output, lstm_hidden, cell



    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

        if torch.cuda.is_available():
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden























#V2
# class RecurrentMN(nn.Module):
#     def __init__(self, hidden_size, output_size,  memory_size, n_layers=1):
#         super(RecurrentMN, self).__init__()
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size) 
#         self.lstm = nn.LSTM(hidden_size, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size, hidden_size, n_layers)
#         self.gru = nn.GRU(hidden_size*2, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size*2, hidden_size)
#         self.out = nn.Linear(hidden_size*2, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size*2, output_size)
#         self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
#         ############## Memory Block ##################################################
#         self.m_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
#         self.c_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)

#         self.mb_attn_linear = nn.Linear(1, memory_size)

#         self.mb_gru = nn.GRU(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size , hidden_size)
#         self.second_lstm = nn.LSTM(hidden_size, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size, hidden_size, n_layers)
#         ############################################################################## 


#     def forward(self, input, context, batch_size, memory_tensor, memory_size): #hidden should be renamed as context

#         ############## Memory Block ####################################################

#         input_memory = self.m_embedding(memory_tensor) #M 
#         output_memory = self.c_embedding(memory_tensor) #C

#         attn = torch.bmm(input_memory.permute(1,0,2), context.permute(1,2,0))# batch_size X n X 1
#         attn = attn.squeeze(2)

#         # self.mb_attn_linear = nn.Linear(len(memory_tensor), len(memory_tensor))
#         # attn = self.mb_attn_linear(attn)
#         attn_dist = self.softmax(attn).unsqueeze(2)
#         output_memory = output_memory.permute(1,2,0)

#         memory_context = torch.bmm(output_memory, attn_dist).permute(2,0,1) #S

#         mb_output, mb_hidden = self.mb_gru(memory_context, context)
#         output = mb_hidden

#         ####################################################################################
#         output = torch.cat((output, context),2)
#         output = self.softmax(self.out(output[0]))

#         return output, context
        


#     def init_hidden(self, batch_size):
#         hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

#         if torch.cuda.is_available():
#             hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
#         return hidden



















#V1
# class RecurrentMN(nn.Module):
#     def __init__(self, hidden_size, output_size,  memory_size, n_layers=1):
#         super(RecurrentMN, self).__init__()
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size) 
#         self.lstm = nn.LSTM(hidden_size*2, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size*2, hidden_size, n_layers)
#         self.gru = nn.GRU(hidden_size*2, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size*2, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
#         ############## Memory Block ##################################################
#         self.m_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
#         self.c_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)

#         self.mb_attn_linear = nn.Linear(1, memory_size)

#         self.mb_gru = nn.GRU(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size, hidden_size)
#         self.second_lstm = nn.LSTM(hidden_size, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size, hidden_size, n_layers)
#         ############################################################################## 


#     def forward(self, input, context, batch_size, memory_tensor, memory_size): #hidden should be renamed as context
#         output = self.embedding(input)
#         context = context.view(self.n_layers,batch_size,self.hidden_size)
#         output = F.relu(output)
#         output =  torch.cat((output, context), 2)

#         cell = Variable(torch.zeros(1, batch_size, self.hidden_size))
#         lstm_output, (lstm_hidden, cell) = self.lstm(output, (context, cell))
#         #lstm_output, lstm_hidden = self.gru(output, context)
#         ############## Memory Block ####################################################

#         input_memory = self.m_embedding(memory_tensor) #M 
#         output_memory = self.c_embedding(memory_tensor) #C

#         attn = torch.bmm(input_memory.permute(1,0,2), lstm_hidden.permute(1,2,0))# batch_size X n X 1
#         attn = attn.squeeze(2)

#         self.mb_attn_linear = nn.Linear(len(memory_tensor), len(memory_tensor))
#         attn = self.mb_attn_linear(attn)
#         attn_dist = self.softmax(attn).unsqueeze(2)
#         output_memory = output_memory.permute(1,2,0)

#         memory_context = torch.bmm(output_memory,  attn_dist).permute(2,0,1)

#         #_, mb_output = self.mb_gru(memory_context, context)
#         mb_output = memory_context + lstm_hidden

#         output = mb_output
#         ####################################################################################

#         output = self.softmax(self.out(output[0]))
#         # output = self.softmax(self.out(output[0]))


#         # return output, hidden
#         return output, lstm_hidden



#     def init_hidden(self, batch_size):
#         hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

#         if torch.cuda.is_available():
#             hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
#         return hidden




