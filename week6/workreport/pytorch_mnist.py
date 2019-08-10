#!/usr/bin/env python
# coding: utf-8

# In[59]:


import torch
from torchvision import datasets,transforms
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# In[60]:


# 查看pytorch版本
print(torch.__version__)


# In[61]:


# 使用以下的类库对载入的数据进行变换
transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x.repeat(3,1,1)),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
print(transform)


# In[62]:


# train=True即赋值为训练集，False就是测试集 需要一段运行时间
data_train=datasets.MNIST(root="./data",transform=transform,train=True,download=True)
data_test=datasets.MNIST(root="./data",transform=transform,train=False)
data_train
data_test


# In[63]:


# 据集下载完成后。就要对数据进行装载，利用batch _size来确认每个包的大小，用Shuffle来确认打乱数据集的顺序。
data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=64,shuffle=True)
print(data_loader_train)

data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=64,shuffle=True)


# In[64]:


# 预览
images,labels=next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)

img=img.numpy().transpose(1,2,0)

std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]

img=img*std+mean

print([labels[i] for i in range(4)])

plt.show()


# In[65]:


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=torch.nn.Sequential(torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense=torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(1024,10))
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128) # 在向前传播过程中进行x.view(-1,14*14*128)对参数实现扁平化
        x = self.dense(x)
        return x


# In[66]:


model = Model()
if torch.cuda.is_available():
    model.cuda() # 将所有的模型移动到GPU上
cost = torch.nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters())


# In[67]:


print(model)


# In[68]:


# 卷积神经网络进行模型训练和参数优化，需要几个小时的运行时间
n_epochs = 5
for epoch in range(n_epochs):
    running_loss=0.0
    running_correct=0
    print("Epoch{}/{}".format(epoch,n_epochs))
    print("-"*10)
    for data in data_loader_train:
        print("train ing -start")
        X_train,y_train = data
        # 有GPU加这一行，没有不用加
        # X_train,y_train = X_train.cuda(),y_train.cuda()
        X_train,y_train = Variable(X_train),Variable(y_train)
#         print(X_train)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data,1)
        optimzer.zero_grad()
        loss = cost(outputs,y_train)
        
        loss.backward()
        optimzer.step()
#         running_loss += loss.data[0]
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
        print(running_loss)
        print("train ing -end")
    print("第一个循环结束")
    print(running_loss)
    testing_correct = 0
    for data in data_loader_test:
        X_test,y_test = data
        # 有GPU加下面这一行，没有的不用加
        X_test,y_test = Variable(X_test),Variable(y_test)
        outputs = model(X_test)
        _,pred = torch.max(outputs,1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
          100*running_correct/len(data_train),100*testing_correct/len(data_test)))


# In[69]:


# 程序运行结果
# Loss is:0.0003,Train Accuracy is:99.0000%,Test Accuracy is:98.0000

