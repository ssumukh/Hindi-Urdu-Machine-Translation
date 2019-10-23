#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('run', 'seq2seq.ipynb')


# In[3]:


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim * 2, dim)
        self.layer2 = nn.Linear(dim, 1)
        
    def forward(self, source_h, target_h, source_mask=None):
        
        num_sentences, num_words = source_h.shape[:-1]
        target_h = target_h.repeat(1, num_words).reshape(num_words * num_sentences, -1)
        mlp_input = torch.cat((target_h, source_h.reshape(num_words * num_sentences, -1)
                              ), dim=1).reshape(num_words * num_sentences, -1)
        
        h1 = torch.tanh(self.layer1(mlp_input))
        h2 = self.layer2(h1)
        
        raw_weights = h2.reshape(num_sentences, num_words)
        
        if source_mask is not None:
            raw_weights = source_mask * raw_weights
        weights = torch.softmax(f.relu(raw_weights), dim=1)
        
        return torch.sum(weights.unsqueeze(-1) * source_h, dim=1)


# ```
# source_h = torch.Tensor(np.random.randint(0, 10, (2, 4, 10)))
# target_h = torch.Tensor(np.random.randint(0, 10, (2, 10)))
# a = Attention(10)
# a(source_h, target_h)
# ```

# In[11]:


class DecoderAttention(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, padding_dim=0, start_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_dim)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.word_predictor = nn.Linear(hidden_dim, num_embeddings)
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
            decoder_input = torch.cat((embeddings, decoder_context.unsqueeze(1)), dim=-1)
            y_t, (h, c) = self.decoder(decoder_input, (h, c))
            orig_shape = y_t.shape
            pred_t = self.word_predictor(y_t.reshape(orig_shape[0]*orig_shape[1], 
                                                     -1)).reshape(*orig_shape[:-1], -1).squeeze(1)
            pred.append(pred_t)
            decoder_context = self.attention(source_h, h[-1], source_mask)
            
            if input is None:
                words_selected = torch.max(pred_t, dim=1)[1].unsqueeze(1)
            else:
                words_selected = teacher_words[i].unsqueeze(1)


        pred = torch.stack(pred, dim=1)
        return torch.log_softmax(pred, dim=-1), None


# In[28]:


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, source_vocab_len, target_vocab_len, embedding_dim, hidden_dim, num_layers=1,
                       padding_dim_source=0, start_dim_target=1, padding_dim_target=0):
        super().__init__()
        self.encoder = Encoder(source_vocab_len, embedding_dim, hidden_dim, num_layers, padding_dim_source)
        self.decoder = DecoderAttention(target_vocab_len, embedding_dim, hidden_dim, num_layers, padding_dim_target, start_dim_target)
        
    def forward(self, source_input, target_input=None, source_mask=None):
        output, h, c = self.encoder(source_input)
        return self.decoder(output, h, c, target_input, source_mask=source_mask)


# ```
# s = Seq2SeqWithAttention(10, 12, 9, 8)
# s.cuda()
# input = torch.cuda.LongTensor(np.random.randint(0, 10, (3, 6)))
# target = torch.cuda.LongTensor(np.random.randint(0, 12, (3, 10))
# 
# # without teacher forcing
# s(input)
# 
# # with teacher forcing
# ```
