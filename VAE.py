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


BS=24
d1=smd.run_sim()
d2=d1
datalist=random.sample(d2,len(d2))
length_d=round(4*len(datalist)/4)
data_train=datalist[0:length_d]
data_test=d2[0:length_d]
data_train2=DataLoader(data_train[:math.floor(len(data_train)/BS)*BS],batch_size=BS)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)        

    def forward(self, x, edge_index):     
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

## This decoder takes us from latent space back to physical scene ##
class DecoderMLP(torch.nn.Module):
    def __init__(self):
        super(DecoderMLP, self).__init__()
        self.mlp1=MLP([6, 8, 9],batch_norm=False)

    def forward(self,z):
        x= self.mlp1(z)
        return x
 

from torch.utils.tensorboard import SummaryWriter

out_channels = 2
num_features = data_train[0].num_features
epochs = 1000
loss_in = torch.nn.MSELoss()
sigL=torch.nn.Sigmoid()
model = VGAE(encoder=VariationalGCNEncoder(num_features, out_channels),decoder=DecoderMLP())  # new line
device = torch.device('cpu')

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train():
    model.train()

    for i in data_train2:
        optimizer.zero_grad()
        z = model.encode(i.x,i.edge_index)
        xt2 = model.decode(torch.reshape(z,(BS,6)))
        xt0=xt2[:,1:10]
        loss = loss_in(xt0,torch.reshape(i.y[:,:3],(xt0.size())))

        loss = loss + ((1 / i.num_nodes) * model.kl_loss())/100  # new line 
  
        loss.backward()
        optimizer.step()

    return float(loss)


def test():
    with torch.no_grad():
        model.eval()
        loss=0
        # xout=np.zeros((1,9))
        # xin=np.zeros((1,12))
        # edge_out=np.zeros((1,3))
        for i in data_test:
            optimizer.zero_grad()
            z = model.encode(i.x,i.edge_index)
            xt2 = model.decode(torch.reshape(z,(1,6)))
            xt0=xt2[:,1:10]
            xt1=sigL(xt2[:,9:12])
            loss = loss+loss_in(xt0,torch.reshape(i.y[:,:3],(xt0.size())))

            xi = torch.reshape(i.y,(xt2.size()))
            if i==0:
                xout[0,:]=xt0.detach().numpy()
                xin[0,:]=xi.detach().numpy()
                edge_out[0,:]=xt1.detach().numpy()
            else:
                xout=np.append(xout,xt0.detach().numpy(),axis=0)
                xin=np.append(xin,xi.detach().numpy(),axis=0)
                edge_out=np.append(edge_out,xt1.detach().numpy(),axis=0)

    return loss

writer = SummaryWriter('runs/VGAE_experiment_'+'2d_20_epochs')

for epoch in range(1, epochs + 1):

    loss = train()
    errors = test()

    print('Epoch: {:03d}, AUC: {:.4f}'.format(epoch, errors))
    
fname="./modelFE"+str(epoch)
torch.save(model.state_dict(), fname)
