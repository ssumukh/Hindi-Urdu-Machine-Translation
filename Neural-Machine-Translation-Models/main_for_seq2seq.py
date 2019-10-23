#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'utils.ipynb')


# In[2]:


get_ipython().run_line_magic('run', 'seq2seq.ipynb')


# In[3]:


torch.cuda.set_device(0)


# In[4]:


d = Dataset(source_files, target_files, batch_size=10)
embedding_dim = 50
hidden_dim = 100
num_layers = 2


# In[5]:


print(d)


# **Training**

# In[5]:


m = Seq2Seq(d.len_source, d.len_target, embedding_dim, hidden_dim, num_layers=2, 
            padding_dim_source=d.source_pad, padding_dim_target=d.target_pad, start_dim_target=d.target_start)
print(m.cuda())


# In[7]:


prev_score = -1
epochs = 25
for i in range(epochs):
    train(m, d)
    score = eval(m, d)
    print()
    print(score)
    if score > prev_score:
        torch.save(m.state_dict(), "seq2seq_" + ".params")
        prev_score = score


# In[ ]:




