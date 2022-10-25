import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import pandas as pd
import os
from torch_geometric.nn import VGAE, MLP
import numpy as np
import matplotlib.pyplot as plt

datalist=torch.load('data_1.pt')

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

def write_csv(filename):
    model.eval()
    loss=0
    yout=[]
    xout=[]
    A0=[]
    for i in datalist:
        u=i.edge_attribute

        ## Calculate z_t ##
        z = model.encode(i.x,i.edge_index)
        xt, lin_t = model.decode(torch.reshape(z,(1,6)))

        ## Form A, B and o matrices ##
        A=torch.reshape(lin_t[:,:36],(1,6,6))
        B=torch.reshape(lin_t[:,36:42],(1,6,1))
        o=torch.reshape(lin_t[:,42:],(1,6,1))
        # A0.append(np.linalg.eig(np.reshape(A.detach().numpy(),(6,6))))

        ## Calcutate z_t+1_tilde ##
        zout1=torch.empty(1,6,requires_grad=False)
        z2=torch.reshape(z,(6,1))
        for j in range(1):
            zout1[j,:]=torch.reshape(torch.reshape(A[j,:,:]@z2[:,0],(6,1))+B[j,:]*u[j]+o[j,:],(1,6))
            

        ## Calculate z_t+1 ##
        z1 = model.encode(i.y,i.edge_index)
        xt1, _ = model.decode(torch.reshape(z1,(1,6)))  
        if len(xout)==0:
            xout=xt1.detach().numpy().flatten()
            yout=i.y.detach().numpy().flatten()
            zout=zout1.detach().numpy().flatten()
        else:
            yout=np.vstack((yout,i.y.detach().numpy().flatten()))
            xout=np.vstack((xout,xt1.detach().numpy().flatten()))
            zout=np.vstack((zout,zout1.detach().numpy().flatten()))

        # loss = loss+loss_in(z1.flatten(),zout.flatten())
        # loss = loss+loss_in(xt,torch.reshape(i.x,(xt.size())))

        # loss = loss + ((1 / i.num_nodes) * model.kl_loss())/100  # new line 
    # np.savetxt('zout.txt',zout)
    # np.savetxt('xout.txt',xout)
    # np.savetxt('yout.txt',yout)
    print("hey")


out_channels = 2
num_features = datalist[0].num_features
epochs = 10

loss_in = torch.nn.MSELoss()
loss_edge = torch.nn.BCELoss()
sigL=torch.nn.Sigmoid()

model = VGAE(encoder=VariationalGCNEncoder(num_features, out_channels),decoder=DecoderMLP())  # new line
model.load_state_dict(torch.load("./modelF1000"))
# model.eval()
write_csv('test4.txt')