#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 2021/9/28 13:34
Desc    : 使用GAN生成信道转移概率CSI
"""
import torch

"""
发送长度为K的导频符号序列，其中发送符号x[i]从发送集S中等概率随机选取，接收端接受y[i]
训练数据即T[i]=(S[i],y[i])
收集多个i时刻的信道输入状态向量与信道输出（K个），从而得到训练数据集T=[T1,T2,...Ti]
"""


"""
Step1. 开始构建数据生成器，产生S[i]与y[i]的数据对
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NumbersDataset(Dataset):
    def __init__(self, low, high):
        self.samples = list(range(low, high))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        transmitter = self.samples[idx]
        receiver = transmitter + 1
        return transmitter, receiver

"""
Step2. 开始构建两个全连接神经网络D和G，使参数满足标准高斯分布
所述的生成器G是一个全连接神经网络
"""
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class G(nn.Module):
    def __init__(self, p):
        super(G, self).__init__()
        self.FC1 = nn.Sequential(nn.Linear(4,256), nn.LeakyReLU())
        self.FC2 = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU())
        self.FC3 = nn.Sequential(nn.Linear(512,1024), nn.LeakyReLU())
        self.FC4 = nn.Linear(1024,p)
    def forward(self, s):
#         隐含层层数为Ng
        s = self.FC1(s)
        s = self.FC2(s)
        s = self.FC3(s)
        s = self.FC4(s)
        return nn.Softmax(s)

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.FC1a = nn.Sequential(nn.Linear(10,256), nn.LeakyReLU())
        self.FC1b = nn.Sequential(nn.Linear(4,256), nn.LeakyReLU())
        self.FC2 = nn.Sequential(nn.Linear(512,512), nn.LeakyReLU())
        self.FC3 = nn.Sequential(nn.Linear(512,256), nn.LeakyReLU())
        self.FC4 = nn.Linear(256,1)

    def forward(self, q, qt, s):
        s1a = self.FC1a(q)
        s1b = self.FC1b(s)
        s = torch.cat((s1a,s1b),1)
        s = self.FC2(s)
        s = self.FC3(s)
        s = self.FC4(s)
        return nn.Sigmoid(s)

'''
Step3. 对两个网络进行初始化，按照高斯分布初始化参数结构
'''

# 初始化函数接受一个初始化过的网络作为参数输入，将其参数重新初始化为高斯分布
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

NetG = G(4)
NetD = D()

NetG.apply(weights_init)
NetD.apply(weights_init)

'''
Step4. 分别定义两个网络的损失函数和优化方法，进行训练
'''
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# 这里产生一个固定的噪声，每轮学习完成后都使用固定噪声来进行输出
# fixed_noise = torch.randn(25, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
lr = 0.0002
beta1 = 0.5
optimizerD = torch.optim.Adam(NetD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=lr, betas=(beta1, 0.999))

'''
Step5. 开始进行网络训练
首先对鉴别器进行训练，然后训练生成器
'''
# Training Loop

num_epochs = 100

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
dataset = NumbersDataset(0, 50)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        NetD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = NetD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = NetG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = NetD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        NetG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = NetD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

import numpy as np
if __name__ == '__main__':
    dataset = NumbersDataset(0,50)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(len(dataset))
    print(dataset[5])
    print(next(iter(dataloader)))
    g = G(4)
    d = D()
    a = np.array(1,2,3,4)

    print(g.forward(t))



