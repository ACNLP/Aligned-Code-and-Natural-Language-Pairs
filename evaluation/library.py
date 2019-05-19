import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

import itertools
import string
import joblib
import sys
import os
import re

import nltk

def proc(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return [i for i in x if i > 1]

def bleu(ref, hyp):
    ref = proc(ref)
    hyp = proc(hyp)
    l = min(len(ref), len(hyp))
    if l == 0:
        return 0
    elif 0 < l < 4:
        weights = [1 / l] * l
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=weights)

def shuffled(a):
    i = np.arange(len(a))
    np.random.shuffle(i)
    return a[i]

def pprint(vocab, tokens):
    tokens = tokens.cpu().numpy()
    print(' '.join(vocab[t] for t in tokens if t > 1))

class Encoder(nn.Module):
    def __init__(self, n_emb, n_enc, **p):
        super().__init__()
        self.emb = nn.Embedding(n_emb, n_enc)
        self.enc = nn.GRU(n_enc, n_enc, batch_first=True, **p)
        self.lin = nn.Linear(n_enc, n_emb)
        self.h = None
    
    def set_state(self, h):
        self.h = h
    
    def forward(self, x):
        x, h = self.enc(self.emb(x), self.h)
        self.h = h.detach()
        return x
    
    def encode(self, x):
        x = self.forward(x)
        x = (x.mean(1) + x.max(1)[0] + x[:, -1]) / 3
        return x
    
    def loss(self, x):
        return F.cross_entropy(self.predict(x[:, :-1]).transpose(2, 1), x[:, 1:], reduction='none')
    
    def predict(self, x):
        return self.lin(self.forward(x))

class Matcher(nn.Module):
    def __init__(self, src_enc, dst_enc, n_enc, n_hid):
        super().__init__()
        self.src_enc = src_enc
        self.dst_enc = dst_enc
        self.matcher = nn.Sequential(
            nn.Linear(n_enc * 3, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, 1),
        )
    
    def forward(self, x, y):
        self.src_enc.set_state(None)
        self.dst_enc.set_state(None)
        x = self.src_enc.encode(x)
        y = self.dst_enc.encode(y)
        z = torch.cat([x, y, x * y], 1)
        return self.matcher(z)
    
    def loss(self, x, y, match):
        return F.binary_cross_entropy_with_logits(self.forward(x, y), match, reduction='none')

class Seq2Seq(nn.Module):
    def __init__(self, src_enc, dst_enc):
        super().__init__()
        self.src_enc = src_enc
        self.dst_enc = dst_enc
    
    def forward(self, x):
        self.src_enc.set_state(None)
        self.dst_enc.set_state( self.src_enc.encode(x).unsqueeze(0) )
    
    def loss(self, x, y, _):
        self.forward(x)
        return self.dst_enc.loss(y)
    
    def translate(self, x, maxlen):
        self.forward(x)
        ys = [torch.ones(len(x), 1, dtype=torch.long).to(x.device)]
        for _ in range(1, maxlen):
            ys.append( self.dst_enc.predict(ys[-1]).argmax(-1) )
        return torch.cat(ys[1:], 1)

def pad(xs, maxlen, pad_left):
    X = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        if len(x) > 0:
            l = min(len(x), maxlen)
            if pad_left:
                X[i, -l:] = torch.LongTensor(x)[-l:]
            else:
                X[i, :l] = torch.LongTensor(x)[:l]
    return X

class LanguagePairDataset():
    def __init__(s, bundle, src_maxlen, dst_maxlen, use_cuda):
        s.negative_sample = 0 # 0 by default; set to +ve num when training matcher
        s.src_maxlen = src_maxlen
        s.dst_maxlen = dst_maxlen
        s.use_cuda = use_cuda
        
        s.df, s.src_buffer, s.dst_buffer, s.src_vocab, s.dst_vocab = joblib.load(bundle)
        
        s.train = np.where(~s.df.is_test.values)[0]
        s.test = np.where(s.df.is_test.values)[0]
        s.all = np.arange(len(s.df))
    
    def print_sample(self, indices):
        s = self.df.iloc[indices].sample()
        print('=' * 80)
        print()
        print(' '.join(self.src_buffer[ sample['src_s'] : sample['src_e'] ]))
        print()
        print(' '.join(self.dst_buffer[ sample['dst_s'] : sample['dst_e'] ]))
        print()
        print('=' * 80)
    
    def batch(self, indices):
        xs, ys = [], []
        for _, e in self.df.iloc[indices].iterrows():
            # 1: /*Start*/
            xs.append([1] + list(self.src_buffer[ e['src_s'] : e['src_e'] ]))
            ys.append([1] + list(self.dst_buffer[ e['dst_s'] : e['dst_e'] ]))

        ns = int( len(ys) * self.negative_sample )
        if ns > 1:
            y0 = ys[0]
            for i in range(ns):
                ys[i] = ys[i + 1]
            ys[ns - 1] = y0

        x, y, m = (
            pad(xs, maxlen=self.src_maxlen, pad_left=True),
            pad(ys, maxlen=self.dst_maxlen, pad_left=False),
            torch.cat([torch.zeros(ns), torch.ones(len(ys) - ns)]).view(-1, 1),
        )

        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
            m = m.cuda()

        return x, y, m
    
    def one_epoch(self, model, indices, bs, optim=None):
        train = optim is not None
        if train:
            model.train()
        else:
            model.eval()
        ret = torch.zeros(len(indices))
        tot_loss, n_loss = 0, 0
        with tqdm(total=len(indices)) as pbar:
            for i in range(0, len(indices), bs):
                indx = indices[i : i + bs]
                x, y, m = self.batch(indx)
                losses = model.loss(x, y, m)
                for j, l in zip(range(i, i + bs), losses):
                    ret[j] = float(l.mean())
                loss = losses.mean()
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                tot_loss += float(loss) * len(indx)
                n_loss += len(indx)
                pbar.update(len(indx))
                pbar.set_description('loss: %.4f' % (tot_loss / n_loss))
        return ret
