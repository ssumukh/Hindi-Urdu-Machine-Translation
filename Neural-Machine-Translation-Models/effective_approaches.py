#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', 'seq2seq.ipynb')


# In[ ]:


class Attention_General(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)
        
    def forward(self, source_h, target_h, source_mask=None):
        '''
        # TODO: repeat copies data, use expand
        
        source_h => (num_sentences * num_words * embedding_dim)
        target_h => (num_sentences * embedding_dim) 
        '''
        num_sentences, num_words = source_h.shape[:-1]
        target_h = target_h.repeat(1, num_words).reshape(num_words * num_sentences, -1)
        raw_weights = self.bilinear(target_h, source_h.reshape(num_words * num_sentences, 
                                                               -1)).reshape(num_sentences, num_words)
        
        if source_mask is not None:
            raw_weights = source_mask * raw_weights

        weights = torch.softmax(f.relu(raw_weights), dim=1)
        
        return torch.sum(weights.unsqueeze(-1) * source_h, dim=1), weights


# In[ ]:


class Attention_Dot(nn.Module):
    def __init__(self, _):
        super().__init__()
        
    def forward(self, source_h, target_h, source_mask=None):
        num_sentences, num_words = source_h.shape[:-1]
        target_h = target_h.repeat(1, num_words).reshape(num_words * num_sentences, -1)
        raw_weights = torch.sum(target_h * source_h.reshape(num_words * num_sentences, -1), dim=1).reshape(num_sentences, num_words)
        
        if source_mask is not None:
            raw_weights = source_mask * raw_weights
            
        weights = torch.softmax(f.relu(raw_weights), dim=1)
        return torch.sum(weights.unsqueeze(-1) * source_h, dim=1), weights


# In[ ]:


class Attention_Concat(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim*2, 1)
        
    def forward(self, source_h, target_h, source_mask=None):
        num_sentences, num_words = source_h.shape[:-1]
        target_h = target_h.repeat(1, num_words).reshape(num_words * num_sentences, -1)
        
        input = torch.cat((target_h, source_h.reshape(num_words * num_sentences, -1)), dim=1)
        raw_weights = self.w(input).reshape(num_sentences, num_words)
        
        if source_mask is not None:
            raw_weights = source_mask * raw_weights
            
        weights = torch.softmax(f.relu(raw_weights), dim=1)
        return torch.sum(weights.unsqueeze(-1) * source_h, dim=1), weights


# ```
# source_h = torch.Tensor(np.random.randint(0, 10, (2, 4, 10)))
# target_h = torch.Tensor(np.random.randint(0, 10, (2, 10)))
# a = Attention_Concat(10)
# a(source_h, target_h)
# ```

# In[ ]:


class DecoderEffective(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, 
                 padding_dim=0, start_dim=1, Attention=Attention_General):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.word_predictor = nn.Linear(hidden_dim * 2, num_embeddings)
        self.start_dim = start_dim
        
        self.attention = Attention(hidden_dim)
        
    def forward(self, output, h, c, input=None, max_sen_len=20, source_mask=None):
        
        source_h = output
        num_sentences, num_words, hidden_dim = output.shape
        words_selected = torch.cuda.LongTensor([[self.start_dim] for _ in range(num_sentences)])

        decoder_context = torch.zeros(num_sentences, hidden_dim).cuda()
        pred = []
        
        if input is not None:
            max_sen_len = input.shape[1]
            teacher_words = input.t()
        
        for i in range(max_sen_len):
            embeddings = self.embedding(words_selected)
            h_t, (h, c) = self.decoder(embeddings, (h, c))
            orig_shape = h_t.shape
            
            ct, _ = self.attention(source_h, h[-1], source_mask)
            h_t_ = torch.cat((ct, h[-1]), dim=1)
            
            pred_t = self.word_predictor(h_t_.reshape(orig_shape[0]*orig_shape[1], 
                                                     -1)).reshape(*orig_shape[:-1], -1).squeeze(1)
            pred.append(pred_t)
            decoder_context = self.attention(source_h, h[-1])
            
            if input is None:
                words_selected = torch.max(pred_t, dim=1)[1].unsqueeze(1)
            else:
                words_selected = teacher_words[i].unsqueeze(1)


        pred = torch.stack(pred, dim=1)
        return torch.log_softmax(pred, dim=-1), None


# In[ ]:


class Seq2Seq_EffectiveApproaches(nn.Module):
    def __init__(self, source_vocab_len, target_vocab_len, embedding_dim, hidden_dim, num_layers=2,
                       padding_dim_source=0, start_dim_target=1, padding_dim_target=0, Attention=Attention_General):
        super().__init__()
        self.encoder = Encoder(source_vocab_len, embedding_dim, hidden_dim, num_layers, padding_dim_source)
        self.decoder = DecoderEffective(target_vocab_len, embedding_dim, hidden_dim, num_layers, 
                                        padding_dim_target, start_dim_target, Attention=Attention)
        
    def forward(self, source_input, target_input=None, source_mask=None):
        output, h, c = self.encoder(source_input)
        return self.decoder(output, h, c, target_input, source_mask=source_mask)


# ```
# s = Seq2Seq_EffectiveApproaches(10, 12, 9, 8, Attention=Attention_Concat)
# s.cuda()
# input = torch.cuda.LongTensor(np.random.randint(0, 10, (3, 6)))
# target = torch.cuda.LongTensor(np.random.randint(0, 12, (3, 10)))
# 
# # without teacher forcing
# s(input)
# 
# # with teacher forcing
# s(input, target)
# ```
