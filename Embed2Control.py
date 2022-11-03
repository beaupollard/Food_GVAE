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
import time

BS=300
percent_train=0.9
d1=smd.run_sim(run_nums=5,out_data=3)

latent_multi=1.

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
        self.mlp1=MLP([latent_dim, 8, 9],batch_norm=False)
        self.mlp_lin=MLP([latent_dim,24,latent_dim**2+2*latent_dim])


    def forward(self,z):
        x= self.mlp1(z)
        lin=self.mlp_lin(z)
        return x, lin
 

from torch.utils.tensorboard import SummaryWriter
# device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

out_channels = 2
num_features = data_train.dataset[0].num_features
epochs = 200
loss_in = torch.nn.MSELoss()
latent_dim=out_channels*num_features

model = VGAE(encoder=VariationalGCNEncoder(num_features, out_channels),decoder=DecoderMLP())  # new line
model = model.to(device)
# model.load_state_dict(torch.load("./modelFL4002"))
# device = torch.device('cpu')

learning_rate=0.03
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    model.train()
    latent_loss=0
    encode_loss=0
    kl_loss=0
    L2=0
    for i in data_train:
        optimizer.zero_grad()
        i=i.to(device)
        u=i.edge_attribute

        ## Calculate z_t ##
        z = model.encode(i.x,i.edge_index)
        xt, lin_t = model.decode(torch.reshape(z,(BS,latent_dim)))

        ## Form A, B and o matrices ##
        # A=torch.reshape(lin_t[:,:latent_dim**2],(BS,latent_dim,latent_dim))
        # B=torch.reshape(lin_t[:,latent_dim**2:latent_dim**2+latent_dim],(BS,latent_dim,1))
        # o=torch.reshape(lin_t[:,latent_dim**2+latent_dim:],(BS,latent_dim,1))
        A=torch.reshape(lin_t[0,:latent_dim**2],(1,latent_dim,latent_dim))
        B=torch.reshape(lin_t[0,latent_dim**2:latent_dim**2+latent_dim],(1,latent_dim,1))
        o=torch.reshape(lin_t[0,latent_dim**2+latent_dim:],(1,latent_dim,1))
         ## Calcutate z_t+1_tilde ##
        zout=torch.empty(BS,latent_dim,requires_grad=False).to(device)
        z2=torch.reshape(z,(BS,latent_dim))
        for j in range(BS):
            zout[j,:]=torch.reshape(torch.reshape(A[0,:,:]@z2[j,:],(latent_dim,1))+B[0,:]*u[j]+o[0,:],(1,latent_dim))
            

        ## Calculate z_t+1 ##
        z1 = model.encode(i.y,i.edge_index)
        xt1, _ = model.decode(torch.reshape(z,(BS,latent_dim)))  
        

        ## Loss from z_t+1 estimate ##
        loss = latent_multi*loss_in(z1.flatten(),zout.flatten())

        ## Loss from x_tilde ##
        loss = loss+loss_in(xt,torch.reshape(i.x,(xt.size())))
        
        ## Loss from x_tilde ##
        loss = loss + ((1 / i.num_nodes) * model.kl_loss())/50  # new line check out Soft free bits or KL anneling
        L2=L2+loss
        loss.backward()
        optimizer.step()
        latent_loss+=latent_multi*loss_in(z1.flatten(),zout.flatten())
        encode_loss+=loss_in(xt,torch.reshape(i.x,(xt.size())))
        kl_loss+=((1 / i.num_nodes) * model.kl_loss())/50
    # print(L2)

    return float(loss), 1000*L2, 1000*latent_loss, 1000*encode_loss, 1000*kl_loss


def test():
    with torch.no_grad():
        model.eval()
        loss=0

        for i in data_test:
            optimizer.zero_grad()
            i=i.to(device)
            u=i.edge_attribute

            ## Calculate z_t ##
            z = model.encode(i.x,i.edge_index)
            xt, lin_t = model.decode(torch.reshape(z,(BS2,latent_dim)))

            ## Form A, B and o matrices ##
            # A=torch.reshape(lin_t[:,:latent_dim**2],(BS2,latent_dim,latent_dim))
            # B=torch.reshape(lin_t[:,latent_dim**2:latent_dim**2+latent_dim],(BS2,latent_dim,1))
            # o=torch.reshape(lin_t[:,latent_dim**2+latent_dim:],(BS2,latent_dim,1))
            A=torch.reshape(lin_t[0,:latent_dim**2],(1,latent_dim,latent_dim))
            B=torch.reshape(lin_t[0,latent_dim**2:latent_dim**2+latent_dim],(1,latent_dim,1))
            o=torch.reshape(lin_t[0,latent_dim**2+latent_dim:],(1,latent_dim,1))

            ## Calcutate z_t+1_tilde ##
            zout=torch.empty(BS2,latent_dim,requires_grad=False).to(device)
            z2=torch.reshape(z,(BS2,latent_dim))
            for j in range(BS2):
                zout[j,:]=torch.reshape(torch.reshape(A[0,:,:]@z2[j,:],(latent_dim,1))+B[0,:]*u[j]+o[0,:],(1,latent_dim))
                

            ## Calculate z_t+1 ##
            z1 = model.encode(i.y,i.edge_index)
            xt1, _ = model.decode(torch.reshape(z,(BS2,latent_dim)))  

            loss = loss+latent_multi*loss_in(z1.flatten(),zout.flatten())
            loss = loss+loss_in(xt,torch.reshape(i.x,(xt.size())))

            loss = loss + ((1 / i.num_nodes) * model.kl_loss())/100  # new line                 

    return loss

writer = SummaryWriter('runs/VGAE_experiment_'+'2d_20_epochs')
errors=0
count=0
count2=0
t0=time.time()
errout=[]
for epoch in range(1, epochs + 1):

    loss, errors2, latent_loss, encode_loss, KL = train()
    errout.append([latent_loss, encode_loss, KL])
    if count>10:
        errors = test()
        count=0
    count=count+1
    # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    if count2==500:
        learning_rate=learning_rate/2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
        count2=0
    count2+=1

    print('Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}, Latent: {:.4f}, Encode: {:.4f}, KL: {:.4f}'.format(epoch, errors2, errors, latent_loss, encode_loss, KL))
    
    
    # writer.add_scalar('auc train',auc,epoch) # new line
    # writer.add_scalar('ap train',ap,epoch)   # new line
fig, axs = plt.subplots(3, 1)
axs[0].plot(latent_loss[100:],'b')
axs[1].plot(encode_loss[100:],'r')
axs[2].plot(KL[100:],'b')
plt.show()
print(time.time()-t0)
fname="./modelFL"+str(epoch)
torch.save(model.state_dict(), fname)
