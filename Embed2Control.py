from distutils.log import error
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import pandas as pd
import os
from torch_geometric.nn import VGAE, MLP, ECConv
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import springmassdamper as smd
import copy

BS=100
percent_train=0.8
d1=smd.run_sim()

datalist=random.sample(d1,len(d1))
length_d=int(percent_train*len(datalist))
data_train2=datalist[0:length_d]

data_train=DataLoader(data_train2[:math.floor(length_d/BS)*BS],batch_size=BS)
BS2=len(datalist[length_d:])
data_test=DataLoader(datalist[length_d:],batch_size=BS2)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)        

    def forward(self, x, edge_index):     
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class DecoderMLP(torch.nn.Module):
    def __init__(self):
        super(DecoderMLP, self).__init__()
        self.mlp1=MLP([6, 8, 9],batch_norm=False)
        self.mlp_lin=MLP([6,24,48])


    def forward(self,z):
        x= self.mlp1(z)
        lin=self.mlp_lin(z)
        return x, lin
 

from torch.utils.tensorboard import SummaryWriter

out_channels = 2
num_features = data_train.dataset[0].num_features
epochs = 10000
loss_in = torch.nn.MSELoss()

model = VGAE(encoder=VariationalGCNEncoder(num_features, out_channels),decoder=DecoderMLP())  # new line

device = torch.device('cpu')

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()

    L2=0
    for i in data_train:
        optimizer.zero_grad()
        u=i.edge_attribute

        ## Calculate z_t ##
        z = model.encode(i.x,i.edge_index)
        xt, lin_t = model.decode(torch.reshape(z,(BS,6)))

        ## Form A, B and o matrices ##
        A=torch.reshape(lin_t[:,:36],(BS,6,6))
        B=torch.reshape(lin_t[:,36:42],(BS,6,1))
        o=torch.reshape(lin_t[:,42:],(BS,6,1))

         ## Calcutate z_t+1_tilde ##
        zout=torch.empty(BS,6,requires_grad=False)
        z2=torch.reshape(z,(BS,6))
        for j in range(BS):
            zout[j,:]=torch.reshape(torch.reshape(A[j,:,:]@z2[j,:],(6,1))+B[j,:]*u[j]+o[j,:],(1,6))
            

        ## Calculate z_t+1 ##
        z1 = model.encode(i.y,i.edge_index)
        xt1, _ = model.decode(torch.reshape(z,(BS,6)))  

        ## Loss from z_t+1 estimate ##
        loss = loss_in(z1.flatten(),zout.flatten())

        ## Loss from x_tilde ##
        loss = loss+loss_in(xt,torch.reshape(i.x,(xt.size())))
        
        ## Loss from x_tilde ##
        loss = loss + ((1 / i.num_nodes) * model.kl_loss())/100  # new line check out Soft free bits or KL anneling
        L2=L2+loss
        loss.backward()
        optimizer.step()
    # print(L2)

    return float(loss), L2


def test():
    with torch.no_grad():
        model.eval()
        loss=0

        for i in data_test:
            optimizer.zero_grad()
            u=i.edge_attribute

            ## Calculate z_t ##
            z = model.encode(i.x,i.edge_index)
            xt, lin_t = model.decode(torch.reshape(z,(BS2,6)))

            ## Form A, B and o matrices ##
            A=torch.reshape(lin_t[:,:36],(BS2,6,6))
            B=torch.reshape(lin_t[:,36:42],(BS2,6,1))
            o=torch.reshape(lin_t[:,42:],(BS2,6,1))

            ## Calcutate z_t+1_tilde ##
            zout=torch.empty(BS2,6,requires_grad=False)
            z2=torch.reshape(z,(BS2,6))
            for j in range(BS2):
                zout[j,:]=torch.reshape(torch.reshape(A[j,:,:]@z2[j,:],(6,1))+B[j,:]*u[j]+o[j,:],(1,6))
                

            ## Calculate z_t+1 ##
            z1 = model.encode(i.y,i.edge_index)
            xt1, _ = model.decode(torch.reshape(z,(BS2,6)))  

            loss = loss+loss_in(z1.flatten(),zout.flatten())
            loss = loss+loss_in(xt,torch.reshape(i.x,(xt.size())))

            loss = loss + ((1 / i.num_nodes) * model.kl_loss())/100  # new line                 

    return loss

writer = SummaryWriter('runs/VGAE_experiment_'+'2d_20_epochs')
errors=0
count=0
for epoch in range(1, epochs + 1):

    loss, errors2 = train()
    if count>10:
        errors = test()
        count=0
    count=count+1
    # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)

    print('Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, errors2, errors))
    
    
    # writer.add_scalar('auc train',auc,epoch) # new line
    # writer.add_scalar('ap train',ap,epoch)   # new line
fname="./modelFN"+str(epoch)
torch.save(model.state_dict(), fname)
