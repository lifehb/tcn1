#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils import weight_norm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


T = 150
# training epochs
epochs = 1_000
batch_size=90
# temporal casual layer channels
channel_sizes = [10] * 4
lr=0.015
# convolution kernel size
kernel_size = 5
dropout = 0.2


# In[3]:


# 导入数据集
class MyDataset(Dataset):
    def __init__(self, file_path, T):
        self.data = pd.read_csv(file_path)
        self.data.iloc[:,3:-2]=2*(self.data.iloc[:,3:-2]-self.data.iloc[:,3:-2].min())/(self.data.iloc[:,3:-2].max()-self.data.iloc[:,3:-2].min())-1
        self.time = T
        
    def __len__(self):
        return len(self.data) - self.time + 1
    
    def __getitem__(self, index):        
        if index >= len(self.data) - self.time + 1:
            raise IndexError("Index out of range.")
        
        x = torch.tensor(self.data.iloc[index:index+self.time, 3:-2].to_numpy())
        y = torch.tensor(self.data.iloc[index:index+self.time, -1].to_numpy())
        
        return x, y


# In[4]:


mydata=MyDataset("my_data.csv",T)


# In[5]:


train_loader=DataLoader(mydata,batch_size=batch_size,shuffle=True)


# In[6]:


#TCN网络的模型
class Crop(nn.Module):
 
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size
 
    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()


# In[7]:


class TemporalCasualLayer(nn.Module):
 
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout = 0.2):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      stride,
            'padding':     padding,
            'dilation':    dilation
        }
 
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
 
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
 
        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)
        #shortcut connect
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
 
    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)


# In[8]:


class TemporalConvolutionNetwork(nn.Module):
 
    def __init__(self, num_inputs, num_channels, kernel_size = 2, dropout = 0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride':      1,
            'dropout':     dropout
        }
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)


# In[9]:


class TCN(nn.Module):
 
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size = kernel_size, dropout = dropout)
        self.pool1=nn.AdaptiveAvgPool1d(1)
#         self.linear = nn.Linear(num_channels[-1], output_size)
 
    def forward(self, x):
        y = self.tcn(x)#[N,C_out,L_out=L_in]
        y=y.permute(0,2,1)
        y=self.pool1(y)
        return y.squeeze()


# In[10]:


model_params = {
    # 'input_size',C_in
    'input_size':   3,
    # 单步，预测未来一个时刻
    'output_size':  1,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout
}
model = TCN(**model_params)
model=model.cuda()


# In[11]:


optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
mse_loss = torch.nn.MSELoss()


# In[12]:


train_loss=[]


# In[13]:


for t in range(epochs):
#     train_loss=0
    for x,y in train_loader:
        optimizer.zero_grad()
        x=x.cuda()
        y=y.cuda()
        predict_=model(x.permute(0,2,1).float())
        loss=mse_loss(predict_,y.float()).cuda()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
    if (t+1)%5==0:
        print("train_loss=",loss.item())


# In[ ]:




