#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', 'effective_approaches.ipynb')


# In[ ]:


class Decoder_Coverage(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, 
                 padding_dim=0, start_dim=1, Attention=Attention_General):
        
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.word_predictor = nn.Linear(hidden_dim * 2, num_embeddings)
        self.start_dim = start_dim
        
        self.attention = Attention(hidden_dim)        
        self.fertility = nn.Linear(hidden_dim, 1)
        
    def forward(self, output, h, c, input=None, max_sen_len=20, source_mask=None):
        
        source_h = output
        num_sentences, num_words, hidden_dim = output.shape
        words_selected = torch.cuda.LongTensor([[self.start_dim] for _ in range(num_sentences)])

        source_h_importance = self.fertility(source_h.reshape(num_sentences * num_words, -1)).reshape(num_sentences, num_words)
        
        decoder_context = torch.zeros(num_sentences, hidden_dim).cuda()
        pred = []
        attention_weights = []
        
        if input is not None:
            max_sen_len = input.shape[1]
            teacher_words = input.t()
        
        for i in range(max_sen_len):
            embeddings = self.embedding(words_selected)
            h_t, (h, c) = self.decoder(embeddings, (h, c))
            orig_shape = h_t.shape
            
            ct, _attn_weights = self.attention(source_h, h[-1])
            h_t_ = torch.cat((ct, h[-1]), dim=1)
            attention_weights.append(_attn_weights)
            
            pred_t = self.word_predictor(h_t_.reshape(orig_shape[0]*orig_shape[1], 
                                                     -1)).reshape(*orig_shape[:-1], -1).squeeze(1)
            pred.append(pred_t)
            decoder_context = self.attention(source_h, h[-1])
            
            if input is None:
                words_selected = torch.max(pred_t, dim=1)[1].unsqueeze(1)
            else:
                words_selected = teacher_words[i].unsqueeze(1)


        pred = torch.stack(pred, dim=1)
        attention_weights = torch.stack(attention_weights, dim=-1)
        
        return torch.log_softmax(pred, dim=-1), (attention_weights, source_h_importance)


# In[ ]:


class Seq2Seq_Coverage(nn.Module):
    def __init__(self, source_vocab_len, target_vocab_len, embedding_dim, hidden_dim, num_layers=2,
                       padding_dim_source=0, start_dim_target=1, padding_dim_target=0, Attention=Attention_General):
        super().__init__()
        self.encoder = Encoder(source_vocab_len, embedding_dim, hidden_dim, num_layers, padding_dim_source)
        self.decoder = Decoder_Coverage(target_vocab_len, embedding_dim, hidden_dim, num_layers, 
                                        padding_dim_target, start_dim_target, Attention=Attention)
        
    def forward(self, source_input, target_input=None, source_mask=None):
        output, h, c = self.encoder(source_input)
        return self.decoder(output, h, c, target_input)


# ```
# s = Seq2Seq_Coverage(10, 12, 9, 8, Attention=Attention_Concat)
# s.cuda()
# input = torch.cuda.LongTensor(np.random.randint(0, 10, (3, 6)))
# target = torch.cuda.LongTensor(np.random.randint(0, 12, (3, 10)))
# 
# # without teacher forcing
# pred, (attn_weights, fertility) = s(input)
# 
# # with teacher forcing
# pred, (attn_weights, fertility) = s(input, target)
# ```
