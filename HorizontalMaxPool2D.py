#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch.nn as nn

class HorizontalMaxPool2D(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2D,self).__init__()
        
    def forward(self,x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,kernel_size=(1,inp_size[3]))


# In[13]:


if __name__ == '__main__':
    import torch
    x = torch.Tensor(32,2048,8,4)
    hp=HorizontalMaxPool2D()
    y=hp(x)
    print(y.shape)


# In[ ]:




