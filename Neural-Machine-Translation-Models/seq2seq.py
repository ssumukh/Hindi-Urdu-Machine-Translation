#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


# In[3]:


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, padding_dim=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, input):
        '''
        input dim => 2d no_sentences * no_words
        make sure that the input is in reverse
        
        Outputs:
        output, h, c
        
        output => dim: (num_sentences * num_words * hidden_dim)
        h => (num_layers * num_sentences * hidden_dim)
        c => (num_layers * num_sentences * hidden_dim)
        '''
        embeddings = self.embedding(input)
        output, (h, c) = self.encoder(embeddings)
        
        return output, h, c


# In[4]:


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, padding_dim=0, start_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.word_predictor = nn.Linear(hidden_dim, num_embeddings)
        self.start_dim = start_dim
        
    def forward(self, output, h, c, input=None, max_sen_len=20):
        '''
        Output:
        pred: (no_sentences * no_words * no_embeddings)
        **Note: if input is not None, log_softmax is returned
        else, input is returned without softmax**
        '''
        
        if input is not None: # training with teacher forcing
            
            embeddings = self.embedding(input)
            yts, _ = self.decoder(embeddings, (h, c))
            orig_shape = yts.shape
            pred = self.word_predictor(yts.reshape(orig_shape[0]*orig_shape[1], -1)).reshape(*orig_shape[:-1], -1)
            return torch.log_softmax(pred, dim=-1), None
        
        else: # training using the decoders' own predictions
            
            num_sentences = output.shape[0]
            words_selected = torch.cuda.LongTensor([[self.start_dim] for _ in range(num_sentences)])
            pred = []
            for i in range(max_sen_len):
                embeddings = self.embedding(words_selected)
                y_t, (h, c) = self.decoder(embeddings, (h, c))
                orig_shape = y_t.shape
                pred_t = self.word_predictor(y_t.reshape(orig_shape[0]*orig_shape[1], 
                                                         -1)).reshape(*orig_shape[:-1], -1).squeeze(1)
                pred.append(pred_t)
                words_selected = torch.max(pred_t, dim=1)[1].unsqueeze(1)
            
            pred = torch.stack(pred, dim=1)
            return torch.log_softmax(pred, dim=-1), None


# In[5]:


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_len, target_vocab_len, embedding_dim, hidden_dim, num_layers=1,
                       padding_dim_source=0, start_dim_target=1, padding_dim_target=0):
        super().__init__()
        self.encoder = Encoder(source_vocab_len, embedding_dim, hidden_dim, num_layers, padding_dim_source)
        self.decoder = Decoder(target_vocab_len, embedding_dim, hidden_dim, num_layers, padding_dim_target, start_dim_target)
        
    def forward(self, source_input, target_input=None, source_mask=None):
        output, h, c = self.encoder(source_input)
        return self.decoder(output, h, c, target_input)


# **Without teacher forcing**
# ```
# input = torch.cuda.LongTensor(np.random.randint(0, 10, (3, 6)))
# s = Seq2Seq(10, 12, 9, 7)
# s.cuda()
# pred = s(input)
# ```

# **With teacher forcing**
# 
# ```
# tar_input = torch.cuda.LongTensor(np.random.randint(0, 10, (3, 6)))
# s(input, tar_input).shape
# ```

# In[ ]:




