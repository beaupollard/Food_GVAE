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
from scipy import signal
import copy

def apply_filter(xin,N=4,fc=5,dt=0.1):
    fs=1/dt
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    out2=[]
    for i in range(len(xin[0,:])):
        output = signal.filtfilt(b, a, xin[:,i])
        out2.append(output)
    return np.array(out2).T
    

    # return signal.sosfilt(sos,xin)

def ploting(x,y):
    
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        axs[0, i].plot(x[:,i],'b')
        axs[0, i].plot(y[:,i],'r')
        axs[1, i].plot(x[:,i+3],'b')
        axs[1, i].plot(y[:,i+3],'r')
        axs[2, i].plot(x[:,i+6],'b')
        axs[2, i].plot(y[:,i+6],'r')        
    plt.show()

def plot_pos(x,y,index=[0,3,6]):
    plt.rcParams.update({'font.size': 25})
    fig, axs = plt.subplots(3, 1)
    time=np.linspace(0,len(x[:,0])*0.1,len(x[:,0]))
    for i, ind in enumerate(index):
        axs[i].plot(time[:],x[:,ind],'b',linewidth=3)
        axs[i].plot(time[:],y[:,ind],'r',linewidth=3)
        
        axs[i].set_ylabel('Position (m)')
    axs[-1].set_xlabel('Time (s)')     
    plt.show()

def ploting_latent(x,y):
    
    fig, axs = plt.subplots(2, 3)
    for i in range(3):
        axs[0, i].plot(x[:,i])
        axs[0, i].plot(y[:,i])
        axs[1, i].plot(x[:,i+3])
        axs[1, i].plot(y[:,i+3])
    plt.show()


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

def roll_forward(A,B,O,x,edge,u):
    model.eval()
    z0 = model.encode(x,edge)
    xout=np.empty((len(u),9))

    for i in range(1,len(u)):
        zout=torch.tensor(A@z0.detach().numpy().flatten()+(B*u[i]).flatten()+O.flatten()).reshape((3,4))
        xt, lin_t = model.decode(torch.reshape(zout.to(torch.float),(1,latent_dim)))
        xout[i,:]=xt.detach().numpy().flatten()
        z0=copy.deepcopy(zout)
    return xout

def write_csv(filename):
    model.eval()
    loss=0
    yout=[]
    xout=[]
    A0=[]
    B0=[]
    O0=[]
    for i in datalist:
        u=i.edge_attribute

        ## Calculate z_t ##
        z = model.encode(i.x,i.edge_index)
        xt, lin_t = model.decode(torch.reshape(z,(1,latent_dim)))

        ## Form A, B and o matrices ##
        A=torch.reshape(lin_t[0,:latent_dim**2],(1,latent_dim,latent_dim))
        B=torch.reshape(lin_t[0,latent_dim**2:latent_dim**2+latent_dim],(1,latent_dim,1))
        o=torch.reshape(lin_t[0,latent_dim**2+latent_dim:],(1,latent_dim,1))
        # A0.append(np.linalg.eig(np.reshape(A.detach().numpy(),(latent_dim,latent_dim))))
        A0.append((np.reshape(A.detach().numpy(),(1,latent_dim*latent_dim))))
        B0.append((np.reshape(B.detach().numpy(),(1,latent_dim))))
        O0.append((np.reshape(o.detach().numpy(),(1,latent_dim))))

        ## Calcutate z_t+1_tilde ##
        zout1=torch.empty(1,latent_dim,requires_grad=False)
        z2=torch.reshape(z,(latent_dim,1))
        for j in range(1):
            zout1[j,:]=torch.reshape(torch.reshape(A[j,:,:]@z2[:,0],(latent_dim,1))+B[j,:]*u[j]+o[j,:],(1,latent_dim))
            

        ## Calculate z_t+1 ##
        z1 = model.encode(i.y,i.edge_index)
        xt1, _ = model.decode(torch.reshape(z1,(1,latent_dim)))  
        if len(xout)==0:
            xout=xt1.detach().numpy().flatten()
            yout=i.y.detach().numpy().flatten()
            zout=z1.detach().numpy().flatten()
            # zout=zout1.detach().numpy().flatten()
        else:
            yout=np.vstack((yout,i.y.detach().numpy().flatten()))
            xout=np.vstack((xout,xt1.detach().numpy().flatten()))
            zout=np.vstack((zout,z1.detach().numpy().flatten()))

    Aout=np.load('./A3.npy')
    Bout=np.load('./B3.npy')
    Oout=np.load('./O3.npy')

    # Aout=np.zeros(np.size(A0[0]))
    # Bout=np.zeros(np.size(B0[0]))
    # Oout=np.zeros(np.size(O0[0]))
    # for i in range(len(A0)):
    #     Aout=Aout+A0[i]
    #     Bout=Bout+B0[i]
    #     Oout=Oout+O0[i]
    # Aout=np.reshape(Aout,(latent_dim,latent_dim))/len(A0)
    # Bout=np.reshape(Bout,(latent_dim,1))/len(A0)
    # Oout=np.reshape(Oout,(latent_dim,1))/len(A0)
    zout_tilde=[]
    for i in datalist:
        u=i.edge_attribute
        z = model.encode(i.x,i.edge_index)
        ## Calcutate z_t+1_tilde ##
        zout1=torch.empty(1,latent_dim,requires_grad=False)
        z2=torch.reshape(z,(latent_dim,1))
        # for j in range(1):
        zout1=Aout@z2.detach().numpy().flatten()+(Bout)@u.detach().numpy().flatten()+np.reshape(Oout,(latent_dim,))
        if len(zout_tilde)==0:
            zout_tilde=zout1
        else:
            zout_tilde=np.vstack((zout_tilde,zout1))
    # apply_filter(zout)
    # ploting_latent(apply_filter(zout,N=5,fc=1.5),apply_filter(zout_tilde,N=5,fc=1.5))
    # ploting_latent(zout,zout_tilde)
    x=np.empty((len(datalist),9))
    u=np.empty((len(datalist),1))
    for i, data in enumerate(datalist):
        x[i,:]=data.x.detach().numpy().flatten()
        u[i,0]=data.edge_attribute.detach().numpy()
        
    xout_tilde=roll_forward(Aout,Bout,Oout,datalist[1].x,datalist[0].edge_index,u[:])
    print("hey")
    # np.save('./A3',Aout)
    # np.save('./B3',Bout)
    # np.save('./O3',Oout)


datalist=torch.load('data_1.pt')
out_channels = 4
num_features = datalist[0].num_features
epochs = 10
latent_dim=out_channels*num_features

loss_in = torch.nn.MSELoss()
loss_edge = torch.nn.BCELoss()
sigL=torch.nn.Sigmoid()

model = VGAE(encoder=VariationalGCNEncoder(num_features, out_channels),decoder=DecoderMLP())  # new line
model.load_state_dict(torch.load("./modelFL4000"))
# model.eval()
write_csv('test4.txt')