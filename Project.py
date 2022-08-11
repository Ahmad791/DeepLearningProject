import numpy
from torchtext import data
import torch.nn as nn
import torch.utils.data
import spacy
import pandas
from torchtext.data import BucketIterator
from mod import init_mod

def loadData():
    field_args = dict(tokenize='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    include_lengths=True,
                    lower=True) 
    spacy.load('en_core_web_sm')
    tgt_field = data.Field(tokenizer_language="en_core_web_sm", **field_args)
    src_field = data.Field(tokenizer_language="en_core_web_sm", **field_args)
    a,b,c = data.TabularDataset.splits(    path = '.',
                                train='train.csv',
                                validation='validation.csv',
                                test='test.csv',
                                format = 'csv',
                                fields = (('src',src_field), ('trg',tgt_field)))
    src_field.build_vocab(a, min_freq=2)
    tgt_field.build_vocab(a, min_freq=2)
    PAD_TOKEN = src_field.vocab.stoi['<pad>']
    UNK_TOKEN = src_field.vocab.stoi['<unk>']
    dl_train,dl_valid,dl_test = BucketIterator.splits((a,b,c), batch_size = 16)
    #---------------------------end of preparing data
    print('#train samples: ', len(dl_train))
    print('#valid samples: ', len(dl_valid))
    print('#test  samples: ', len(dl_test))
    print('#vocab size: ',len(src_field.vocab),len(tgt_field.vocab))
    return dl_train,dl_valid,dl_test,len(src_field.vocab),len(tgt_field.vocab),(UNK_TOKEN,PAD_TOKEN)
import tqdm
import sys
import matplotlib.pyplot as plt
def train_seq2seq(model, dl_train, optimizer, loss_fn, p_tf=1., clip_grad=1., max_batches=None):
    losses = []
    with tqdm.tqdm(total=(max_batches if max_batches else len(dl_train)), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_train, start=1):
            x, x_len = batch.src
            y, y_len =  batch.trg
            # Forward pass: encoder and decoder
            # Output y_hat is the translated sequence
            y_hat = model(x, y, p_tf, src_len=x_len)
            S, B, V = y_hat.shape
            # y[:,i] is <sos>, w_1, w_2, ..., w_k, <eos>, <pad>, ...
            # y_hat is   w_1', w_2', ..., w_k', <eos>', <pad>', ...
            # based on the above, get ground truth y
            y_gt = y[1:, :].reshape(S*B)  # drop <sos>
            y_hat = y_hat.reshape(S*B, V)
            # Calculate loss compared to ground truth y
            optimizer.zero_grad()
            loss = loss_fn(y_hat, y_gt)
            loss.backward()
            # Prevent large gradients
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            # Update parameters
            optimizer.step()
            losses.append(loss.item())
            pbar.update(); pbar.set_description(f'train loss={losses[-1]:.3f}')
            if max_batches and idx_batch >= max_batches:
                break
    return losses

import numpy as np
def eval_seq2seq(model, dl_test):
    accuracies = []
    with tqdm.tqdm(total=len(dl_test), file=sys.stdout) as pbar:
        for idx_batch, batch in enumerate(dl_test):
            x, x_len = batch.src
            y, y_len =  batch.trg
            with torch.no_grad():
                # Note: no teacher forcing in eval
                y_hat = model(x, y, p_tf=0, src_len=x_len)
            S, B, V = y_hat.shape
            y_gt = y[1:, :] # drop <sos>
            y_hat = torch.argmax(y_hat, dim=2) # greedy-sample (S, B, V) -> (S,B)
            # Compare prediction to ground truth
            accuracies.append(torch.sum(y_gt == y_hat) / float(S))
            pbar.update(); pbar.set_description(f'eval acc={accuracies[-1]}')
    return accuracies






def main():
    dl_train,dl_valid,dl_test,vocabSize,tgtvo,tokens=loadData()
    mod=init_mod(vocabSize,tgtvo)
    optimizer = torch.optim.Adam(mod.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokens[0])
    losses = []
    accuracies = []
    for idx_epoch in range(10):
        p_tf = 1 - min((idx_epoch / 20), 1)    
        print(f'=== EPOCH {idx_epoch+1}/{10}, p_tf={p_tf:.2f} ===')
        losses += train_seq2seq(mod, dl_train, optimizer, loss_fn, p_tf, 1)
        accuracies += eval_seq2seq(mod, dl_valid)


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].plot(losses); ax[0].set_title('train loss'); ax[0].set_xlabel('iteration'); ax[0].grid(True)
    ax[1].plot(accuracies); ax[1].set_title('eval accuracy'); ax[1].set_xlabel('iteration'); ax[1].grid(True)


if __name__=='__main__':
    main()