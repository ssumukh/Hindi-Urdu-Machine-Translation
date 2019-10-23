#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim


# In[4]:


from nltk.translate.bleu_score import sentence_bleu


# In[23]:


data_folder = "data"
files = os.listdir(data_folder)
source_files = [os.path.join(data_folder, file) for file in files if ".ur" in file]
target_files = [os.path.join(data_folder, file) for file in files if ".hn" in file]


# In[24]:


source_files


# In[25]:


target_files


# In[12]:


def generate_vocab(filenames):
    vocab = set()
    no_words = 0
    for file in filenames:
        content = open(file).read()
        sentences = [sentence.split() for sentence in content.split('\n')]
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab.add(word.lower())
                    no_words += 1
    vocab = {word: num for num, word in enumerate(vocab)}
    return vocab


# In[7]:


def parse_files_to_indices(filename, vocab):
    content = open(filename).read()
    return [[vocab[word] for word in sentence.split() if word in vocab] for sentence in content.split('\n')]


# In[13]:


def saveVocabToFl(vocabSet):
    a=list(vocabSet)
    with open('vocabSv.txt', 'w+') as filehandle:  
        for listitem in a:
            filehandle.write('%s\n' % listitem)


# In[16]:


def getSrcVcb():
    vcb = []
    # open file and read the content in a list
    with open('vocabSv.txt', 'r') as filehandle:  
        for line in filehandle:
            # remove linebreak which is the last character of the string
            iii = line[:-1]

            # add item to the list
            vcb.append(iii)
    return set(vcb)


# In[17]:


a={2,3,1,4,5,1,2,1,2,1}
saveVocabToFl(a)
print(getSrcVcb())


# In[18]:


class Dataset:
    def __init__(self, source_files, target_files, batch_size=10):
        self.source_vocab = generate_vocab(source_files)
        saveVocabToFl(self.source_vocab)
        self.target_vocab = generate_vocab(target_files)
        
        self.len_source = len(self.source_vocab.keys())
        self.source_pad, self.source_start, self.source_end = self.len_source + 2, self.len_source + 1, self.len_source
        self.len_target = len(self.target_vocab.keys())
        self.target_pad, self.target_start, self.target_end = self.len_target + 2, self.len_target + 1, self.len_target
        
        self.source_vocab['<pad>'], self.source_vocab['<start>'], self.source_vocab['<end>'] = [self.source_pad, 
                                                                                                self.source_start, 
                                                                                                self.source_end]
        self.target_vocab['<pad>'], self.target_vocab['<start>'], self.target_vocab['<end>'] = [self.target_pad,
                                                                                                self.target_start, 
                                                                                                self.target_end]
        self.len_source = len(self.source_vocab.keys())
        self.len_target = len(self.target_vocab.keys())
        
        self.source_vocab_inv = {value:key for key, value in self.source_vocab.items()}
        self.target_vocab_inv = {value:key for key, value in self.target_vocab.items()}


        for filename in source_files:
            if 'train' in filename:
                self.source_train = parse_files_to_indices(filename, self.source_vocab)
            if 'test' in filename:
                self.source_test = parse_files_to_indices(filename, self.source_vocab)
                
        for filename in target_files:
            if 'train' in filename:
                self.target_train = parse_files_to_indices(filename, self.target_vocab)
            if 'test' in filename:
                self.target_test = parse_files_to_indices(filename, self.target_vocab)
                
            
        self.indices = list(range(len(self.source_train)))
        random.shuffle(self.indices)
        self.current = -batch_size
        self.batch_size = batch_size
        
    def __get_batch_input(self, indices):
        source, target_input, target_target = [], [], []
        
        # padding length
        source_max_len = max(len(self.source_train[i]) for i in indices)
        target_max_len = max(len(self.target_train[i]) for i in indices)
        
        for i in indices:
            length = len(self.source_train[i])
            # reverse_source_sentences and pad at beginning
            sentence = [self.source_pad for _ in range(source_max_len - length)] + [self.source_end] + self.source_train[i][::-1] + [self.source_start]
            source.append(sentence)
            
            length = len(self.target_train[i])
            # padding at end for target
            sentence = [self.target_start] + self.target_train[i] + [self.target_end] + [self.target_pad for _ in range(target_max_len - length)]
            target_input.append(sentence)
            target_target.append(sentence[1:] + [self.target_pad])
            
        return np.array(source), np.array(target_input), np.array(target_target)
    
    def get_batch_input(self):
        if self.current > len(self.indices) - self.batch_size:
            self.current = 0
            return None, None, None
        self.current += self.batch_size
        return self.__get_batch_input(self.indices[self.current: self.current + self.batch_size])
    
    def convert_indices_to_words(self, indices):
        output = []
        for sentence in indices:
            s = []
            for word in sentence:
                s.append(self.target_vocab_inv[word])
            output.append(s)
        return output


# ```
# %run seq2seq.ipynb
# 
# d = Dataset(source_files, target_files, 10)
# m = Seq2Seq(d.len_source, d.len_target, 9, 7)
# m.cuda()
# 
# a, b = d.get_batch_input()
# source, target = torch.cuda.LongTensor(a), torch.cuda.LongTensor(b)
# 
# # For teacher forcing:
# m(source, target)
# 
# # Using model's own predictions:
# m(source)
# ```

# In[9]:


def train(model, dataset, coverage=False, coverage_type="linguistic", iterations=1, use_teacher_forcing=True, log=True):
    # TODO: attention error
    loss_func = nn.NLLLoss()
    attn_loss_func = nn.MSELoss()
    
    optimizer = optim.Adagrad(model.parameters())
    
    for i in range(iterations):
        while True:
            optimizer.zero_grad()
            source, target_input, target_output = d.get_batch_input()

            if source is None: # end of current iteration
                break

            source, target_input, target_output = [torch.cuda.LongTensor(source), 
                                                   torch.cuda.LongTensor(target_input), 
                                                   torch.cuda.LongTensor(target_output)]
            
            source_mask = torch.ones(source.shape).cuda()
            source_mask[source == dataset.source_pad] = 0
            
            if use_teacher_forcing:
                pred, attn = model(source, target_input, source_mask=source_mask)
                # mask whatevers after <stop> 
                target_mask = torch.ones(target_output.shape).cuda()
                target_mask[target_output == dataset.target_pad] = 0
                pred = pred * target_mask.unsqueeze(-1)
                target_output = target_output * target_mask.long()
            else:
                pred, attn_weights = model(source)
                
            no_words = pred.shape[0] * pred.shape[1]
            pred = pred.reshape(no_words, -1)
            target_output = target_output.reshape(no_words)

            pred_error = loss_func(pred, target_output)
            attn_error = None
            
            if coverage:
                # if coverage type is linguistic, ignore fertility
                attn_weights, fertility = attn
                if coverage_type == "linguistic":
                    fertility = torch.ones(fertility.shape).cuda()
                attn_error = attn_loss_func(torch.sum(attn_weights, dim=-1) * source_mask, fertility * source_mask)
                pred_error += attn_error
                
            pred_error.backward()
            optimizer.step()
            
            if log:
                print(d.current/d.batch_size, pred_error, end='\r')


# In[10]:


def convert_pred_to_indices(pred):
    return torch.max(pred, dim=-1)[1]


# In[11]:


def eval(model, dataset, log=False, source_test=None, target_test=None):
    scores = []
    if source_test is None:
        source_test = dataset.source_test
        target_test = dataset.target_test
        
    for sentence, target in zip(source_test, target_test):
        input = torch.cuda.LongTensor([[dataset.source_end] + sentence[::-1] + [dataset.source_start]])
        pred, _ = model(input)
        pred_words = convert_pred_to_indices(pred).cpu().numpy()
        predicted_target = dataset.convert_indices_to_words(pred_words)
        target = dataset.convert_indices_to_words([target])
        if log:
            print(target, predicted_target[0])
        scores.append(sentence_bleu(target, predicted_target[0], weights=(1, 0, 0, 0)))
    return sum(scores)/len(scores)


# In[ ]:




