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
    def __init__(self, input_size, hidden_size, n_layers=2):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, 100).cuda() if torch.cuda.is_available() else nn.Embedding(input_size, 100) 
        self.gru = nn.GRU(100, hidden_size, n_layers).cuda() if torch.cuda.is_available() else nn.GRU(100, hidden_size, n_layers)
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
        self.embedding = nn.Embedding(output_size, 100).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, 100)
        self.gru = nn.GRU(hidden_size + 100, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size+ 100, hidden_size)
        self.out = nn.Linear(hidden_size, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
    def forward(self, input, hidden, context, batch_size): #hidden should be renamed as context

        output = self.embedding(input)
        hidden = hidden.view(self.n_layers,batch_size,self.hidden_size)
        output = F.relu(output)
        output =  torch.cat((output, context), 2)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden




class RecurrentMemory(nn.Module):
    def __init__(self, hidden_size, output_size,  memory_size, n_layers=1):
        super(RecurrentMemory, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, 100).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, 100)
        self.out = nn.Linear(hidden_size*2, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
        ############## Memory Block ##################################################
        self.m_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.c_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.mb_attn_linear = nn.Linear(memory_size, memory_size).cuda() if torch.cuda.is_available() else nn.Linear(memory_size, memory_size)
        self.lstm = nn.LSTM(hidden_size + 100, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size + 100, hidden_size, n_layers)
        self.gru = nn.GRU(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.GRU(hidden_size, hidden_size)
        ############################################################################## 

    def forward(self, input, rmn_hidden, cell,context, batch_size, memory_tensor, memory_size): #hidden should be renamed as context
        output = self.embedding(input)
        context = context.view(self.n_layers,batch_size,self.hidden_size)
        output = F.relu(output)
        output =  torch.cat((output, context), 2)
        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(output, (rmn_hidden, cell))

        ############## Memory Block ####################################################
        input_memory = self.m_embedding(memory_tensor) #M 
        output_memory = self.c_embedding(memory_tensor) #C
        attn = torch.bmm(input_memory.permute(1,0,2), lstm_hidden.permute(1,2,0))# batch_size X n X 1#lstm_hidden.permute(1,2,0)
        attn = attn.squeeze(2)
        attn = self.mb_attn_linear(attn)
        attn_dist = self.softmax(attn).unsqueeze(2)
        output_memory = output_memory.permute(1,2,0)
        memory_context = torch.bmm(output_memory,  attn_dist).permute(2,0,1)
        mb_output,_ = self.gru(memory_context, lstm_hidden)
        ####################################################################################

        mb_output = torch.cat((mb_output, context),2)
        output = self.softmax(self.out(mb_output[0]))
        return output, lstm_hidden, cell


class RMR(nn.Module):
    def __init__(self, hidden_size, output_size,  memory_size, n_layers=1):
        super(RMR, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size) 
        self.lstm = nn.LSTM(hidden_size*2, hidden_size,n_layers).cuda() if torch.cuda.is_available() else nn.LSTM(hidden_size*2, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size*2, output_size).cuda() if torch.cuda.is_available() else nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax().cuda() if torch.cuda.is_available() else nn.LogSoftmax()
        
        ############## Memory Block ##################################################
        self.m_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.c_embedding = nn.Embedding(output_size, hidden_size).cuda() if torch.cuda.is_available() else nn.Embedding(output_size, hidden_size)
        self.mb_attn_linear = nn.Linear(memory_size, memory_size).cuda() if torch.cuda.is_available() else nn.Linear(memory_size, memory_size)
        self.lstm_cell = nn.LSTMCell(hidden_size*2, hidden_size).cuda() if torch.cuda.is_available() else nn.LSTMCell(hidden_size*2, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.GRUCell(hidden_size, hidden_size)        
        ############################################################################## 

        ###### RMR #####
        self.rmr_lstm_cell = nn.LSTMCell(hidden_size, hidden_size).cuda() if torch.cuda.is_available() else nn.LSTMCell(hidden_size, hidden_size)
        ################

    def forward(self, input, rmn_hidden, cell,context, batch_size, memory_tensor, memory_size, rmr_hidden, rmr_cell): #hidden should be renamed as context
        output = self.embedding(input)
        context = context.view(self.n_layers,batch_size,self.hidden_size)
        output = F.relu(output)
        output =  torch.cat((output, context), 2)
        output = output.squeeze(0)
        rmn_hidden = rmn_hidden.squeeze(0)
        cell = cell.squeeze(0)
        lstm_hidden, cell =  self.lstm_cell(output, (rmn_hidden, cell) )
        lstm_hidden = lstm_hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        ############## Memory Block ####################################################
        input_memory = self.m_embedding(memory_tensor) #M 
        output_memory = self.c_embedding(memory_tensor) #C
        attn = torch.bmm(input_memory.permute(1,0,2), lstm_hidden.permute(1,2,0))# batch_size X n X 1#lstm_hidden.permute(1,2,0)
        attn = attn.squeeze(2)
        attn = self.mb_attn_linear(attn)
        attn_dist = self.softmax(attn).unsqueeze(2)
        output_memory = output_memory.permute(1,2,0)
        memory_context = torch.bmm(output_memory,  attn_dist).permute(2,0,1)
        memory_context = memory_context.squeeze(0)
        lstm_hidden = lstm_hidden.squeeze(0)
        mb_output = self.gru_cell(memory_context, lstm_hidden)
        mb_output = mb_output.unsqueeze(0)
        ####################################################################################

        ######################### RMR ################################
        mb_output = mb_output.squeeze(0)
        rmr_hidden = rmr_hidden.squeeze(0)
        rmr_cell = rmr_cell.squeeze(0)

        rmr_hidden, rmr_cell = self.rmr_lstm_cell(mb_output, (rmr_hidden, rmr_cell))

        rmr_hidden = rmr_hidden.unsqueeze(0)
        rmr_cell = rmr_cell.unsqueeze(0)

        rmr_output = torch.cat((rmr_hidden,context),2)
        rmr_output = self.softmax(self.out(rmr_output[0]))
        ########################################################################
        return output, lstm_hidden, cell, rmr_output, rmr_hidden, rmr_cell
