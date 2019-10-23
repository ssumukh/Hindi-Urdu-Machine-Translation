#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'utils.ipynb')


# In[2]:


get_ipython().run_line_magic('run', 'seq2seq_with_attention.ipynb')


# In[3]:


torch.cuda.set_device(0)


# In[4]:


d = Dataset(source_files, target_files, batch_size=10)
embedding_dim = 50
hidden_dim = 100
num_layers = 2


# **Training**

# In[5]:


m = Seq2SeqWithAttention(d.len_source, d.len_target, embedding_dim, hidden_dim, num_layers=2, 
            padding_dim_source=d.source_pad, padding_dim_target=d.target_pad, start_dim_target=d.target_start)
print(m.cuda())


# In[6]:


prev_score = -1
for i in range(20):
    train(m, d)
    score = eval(m, d)
    print()
    print(score)
    if score > prev_score:
        torch.save(m.state_dict(), "attention_" + ".params")
        prev_score = score


# In[ ]:




