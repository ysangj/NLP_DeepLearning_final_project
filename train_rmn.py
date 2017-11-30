import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import gc
import re
import random

import collections
import numpy as np

from torchtext import data
from torchtext import datasets
from model import EncoderRNN, DecoderRNN, RecurrentMemory
import queue
import nltk
from collections import deque


FR = data.Field(init_token='<sos>', eos_token='<eos>')
EN = data.Field(init_token='<sos>', eos_token='<eos>')

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
#'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3
train_set, val_set, test_set = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))

device = 0 if(torch.cuda.is_available()) else -1

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

