from matplotlib import animation
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import random
import math

def run_sim(run_nums=1,out_data=1):
    ## Initialization ##
    tstart = 0
    tend = 100
    dt = 0.1
    t = np.arange(tstart,tend,dt)
    count=1
    int_t = 1
    x0=[]
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    Fiout=[]
    count=1
    xdes=0.5
    Fi=0
    
    ## Integrate using RK4 ##
    def mydiff(x,t,m,k,c,xdes,Fi):
        F=PID(xdes,x[0],Fi)
        dxdt0=x[1]
        dxdt1=1/m[0]*(F+k[1]*(x[2]-x[0])+c[1]*(x[3]-x[1])-k[0]*x[0]-c[0]*x[1])
        dxdt2=x[3]
        dxdt3=1/m[1]*(k[2]*(x[4]-x[2])+c[2]*(x[5]-x[3])-k[1]*(x[2]-x[0])-c[1]*(x[3]-x[1]))
        dxdt4=x[5]
        dxdt5=1/m[2]*(-k[2]*(x[4]-x[2])-c[2]*(x[5]-x[3]))
        dxdt = [dxdt0, dxdt1, dxdt2, dxdt3, dxdt4, dxdt5 ]
        return dxdt

    def PID(xdes,xin,Fi):
        F=-(20*(xin-xdes)+0.0*Fi)
        return F

    ## Run 4 different simulations where each one has different parameters ##
    for j in range(run_nums):

        k=[abs(random.gauss(20.,5.)),abs(random.gauss(20.,5.)),abs(random.gauss(20.,5.))]
        m=[abs(random.gauss(30.,10.)),abs(random.gauss(30.,10.)),abs(random.gauss(30.,10.))]
        c=[abs(random.gauss(10.,5.)),abs(random.gauss(10.,5.)),abs(random.gauss(10.,5.))]
        # x_int=[0.,random.gauss(0.,0.25),0.,random.gauss(0.,0.25),0.,random.gauss(0.,0.25)]
        x_int=[0.,0.,0.,0.,0.,0.]
        omega_des=math.pi/random.gauss(2.,0.5)#0.25*(k[0]/m[0])**0.5
        xd_prev=[]
        for i in range(0,len(t),int_t):
            
            tin=t[i:i+int_t+1]
            xdes=x_int[0]+0.1*math.sin(omega_des*tin[-1])
            xd_prev.append(xdes)
            Fiout = np.append(Fiout,np.ones(len(tin,)-1))
            x = odeint(mydiff, x_int, tin,args=(m,k,c,xdes,Fi))
            Fi=(x[-1,0]-xdes)+Fi

            count=count*-1
            x0=np.append(x0,x[:-1,0])
            x1=np.append(x1,x[:-1,1])
            x2=np.append(x2,x[:-1,2])
            x3=np.append(x3,x[:-1,3])
            x4=np.append(x4,x[:-1,4])
            x5=np.append(x5,x[:-1,5])

            x_int=x[-1,:]


    # ## Recalculate the Forces ##
    # F=[]
    # Fi=0
    # for i,x in enumerate(x4):
    #     F=np.append(F,k[-1]*x+c[-1]*x5[i])
        # F=np.append(F,PID(xdes,x,Fiout[i]))

    ## Add some artificial noise ##
    # nmax=0.0055
    # noise = np.random.normal(0, nmax, x0.shape)
    # x0 = x0 + noise
    # noise = np.random.normal(0, nmax, x0.shape)
    # x1= x1 + noise
    # noise = np.random.normal(0, nmax, x0.shape)
    # x2 = x2 + noise
    # noise = np.random.normal(0, nmax, x0.shape)
    # x3 = x3 + noise
    # noise = np.random.normal(0, nmax, x0.shape)
    # x4 = x4 + noise
    # noise = np.random.normal(0, nmax, x0.shape)
    # x5 = x5 + noise
    # x0=(x0+sum(x0)/len(x0))
    # x0=x0-min(x0)
    # x0=x0/max(x0)
    # x1=(x1+sum(x1)/len(x0))
    # x1=x1-min(x1)
    # x1=x1/max(x1)
    # x2=(x2+sum(x2)/len(x2))
    # x2=x2-min(x2)
    # x2=x2/max(x2)
    # x3=(x3+sum(x3)/len(x2))
    # x3=x3-min(x3)
    # x3=x3/max(x3)
    # x4=(x4+sum(x4)/len(x2))
    # x4=x4-min(x4)
    # x4=x4/max(x4)
    # x5=(x5+sum(x5)/len(x2))
    # x5=x5-min(x5)
    # x5=x5/max(x5)
    ## Calculate the Reaction Forces ##
    F=[]
    Fi=0
    for i,x in enumerate(x4):
        F=np.append(F,k[-1]*x+c[-1]*x5[i])

    ## Save ground truth data to text file ##
    # out_data=1
    zout=np.zeros((len(x0),6))
    for i in range(len(x0)):
        zout[i,:]=np.array([x0[i],x1[i],x2[i],x3[i],x4[i],x5[i]])

    np.savetxt('pos'+str(out_data)+'.txt', zout, delimiter='\t', newline='\n')
    a0=0
    a1=0
    a2=0
    for i in range(len(x0)-1):
        a0=np.append(a0,(x1[i+1]-x1[i])/(dt))
        a1=np.append(a1,(x3[i+1]-x3[i])/(dt))
        a2=np.append(a2,(x5[i+1]-x5[i])/(dt))


    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import os

    ## Save as a datalist ##
    data=[]
    edge=torch.tensor
    count=0
    for i in range(len(x1)-1):

        x=torch.tensor([[x0[i],x1[i],a0[i]],[x2[i],x3[i],a1[i]],[x4[i],x5[i],a2[i]]],dtype=torch.float)
        y=torch.tensor([[x0[i+1],x1[i+1],a0[i+1]],[x2[i+1],x3[i+1],a1[i+1]],[x4[i+1],x5[i+1],a2[i+1]]],dtype=torch.float)
        edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]])
        edge_attribute=torch.tensor([F[i]])    

        data.append((Data(x=x,edge_index=edge_index,y=y,edge_attribute=edge_attribute)))

    torch.save(data,os.path.join('./',f'data_{out_data}.pt'))
    return data

run_sim(run_nums=1)