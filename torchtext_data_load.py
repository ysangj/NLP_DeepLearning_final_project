from torchtext import data
from torchtext import datasets

import re

FR = data.Field()
EN = data.Field()

use_cuda = torch.cuda.is_available()

train, val, test = datasets.IWSLT.splits(exts=('.en', '.fr'), fields=(EN, FR))

FR.build_vocab(train.src, min_freq=1000)
EN.build_vocab(train.trg, min_freq=1000)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=3, device=0)

print(vars(train[10]))
print(EN.vocab.stoi)


batch = next(iter(train_iter))
print(batch.src)
print(batch.trg)
