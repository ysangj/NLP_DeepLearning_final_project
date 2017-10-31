from torchtext import data
from torchtext import datasets

import re

FR = data.Field()
EN = data.Field()

train, val, test = datasets.IWSLT.splits(exts=('.fr', '.en'), fields=(FR, EN))

FR.build_vocab(train.src, min_freq=1000)
EN.build_vocab(train.trg, min_freq=1000)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=3, device=0)

for i in range(len(train) ):
	src = vars(train[i])['src']
	trg = vars(train[i])['trg']
	vars(train[i])['src'] = trg
	vars(train[i])['trg'] = src

for i in range(len(val) ):
	src = vars(val[i])['src']
	trg = vars(val[i])['trg']
	vars(val[i])['src'] = trg
	vars(val[i])['trg'] = src

for i in range(len(test) ):
	src = vars(test[i])['src']
	trg = vars(test[i])['trg']
	vars(test[i])['src'] = trg
	vars(test[i])['trg'] = src



print(EN.vocab.stoi)



# print(vars(train[10]))
# print('\n')
# print(vars(val[10]))
# print('\n')
# print(vars(test[10]))


#Require cuda
batch = next(iter(train_iter))
print(batch.src)
print(batch.trg)
