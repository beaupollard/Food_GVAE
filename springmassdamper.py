from matplotlib import animation
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import random
import math

def normalize(xin):
    meanx=np.mean(xin[-200:])
    maxx=max(xin)
    minx=min(xin)
    xout=(xin-minx)/max(xin-minx)
    xout=xout-np.mean(xout)+meanx
    return xout

def run_sim(run_nums=1,out_data=1,num_repeats=1,test=False):
    ## Initialization ##
    tstart = 0
    tend = 100
    dt = 0.05
    t = np.arange(tstart,tend,dt)
    count=1
    int_t = 1
    x0=[]
    x1=[]
    x0_out=[]
    x1_out=[]
    x2_out=[]
    x3_out=[]
    x4_out=[]
    x5_out=[]    
    F_out=[]
    x4=[]
    x5=[]
    Fiout=[]
    count=1
    xdes=0.5
    Fi=0
    
    ## Integrate using RK4 ##
    def mydiff(x,t,m,k,c,F):
        dxdt0=x[1]
        dxdt1=1/m[0]*(F-k[0]*(x[0]-x[2])-c[0]*(x[1]-x[3]))
        dxdt2=x[3]
        dxdt3=1/m[1]*(k[0]*(x[0]-x[2])+c[0]*(x[1]-x[3])-k[1]*(x[2]-x[4])-c[1]*(x[3]-x[4]))
        dxdt4=x[5]
        dxdt5=1/m[2]*(k[1]*(x[2]-x[4])+c[1]*(x[3]-x[4]))
        dxdt = [dxdt0, dxdt1, dxdt2, dxdt3, dxdt4, dxdt5]
        return dxdt

    def mydiff_force(x,t,m,k,c,xdes,Fi,weight,vdes):
        Frec=(k[0]*x[0]+c[0]*x[1])
        dfdt=Frec-xdes
        F=Frec+(m[0]*(xdes-Frec)-m[0]*dfdt)
        dxdt0=x[1]
        dxdt1=1/m[0]*(F-k[0]*x[0]-c[0]*x[1])
        dxdt = [dxdt0, dxdt1]
        return dxdt

    def PID(xdes,xin,Fi):
        F=-(20*(xin-xdes)+0.0*Fi)
        return F
    w_out=[]
    kout=[15,20,22.5]
    mout=[18,15,25]
    xd_prev=[]
    # k=[abs(random.gauss(20.,5.)),abs(random.gauss(20.,5.)),abs(random.gauss(20.,5.))]
    # m=[abs(random.gauss(30.,5.)),abs(random.gauss(30.,10.)),abs(random.gauss(30.,10.))]
    
    ## Run 4 different simulations where each one has different parameters ##
    for j in range(run_nums):
        # k=[abs(random.gauss(20.,5.))]
        # m=[abs(random.gauss(30.,2.5))]
        k = [20., 15., 17.5]
        m = [26.,30.,20.]
        # k=[kout[j]]#[abs(random.gauss(20.,5.)),abs(random.gauss(20.,5.)),abs(random.gauss(20.,5.))]
        # m=[mout[j]]#[abs(random.gauss(30.,10.)),abs(random.gauss(30.,10.)),abs(random.gauss(30.,10.))]

        c=[2*(k[0]*m[0])**0.5,2*(k[1]*m[1])**0.5*1.05,2*(k[2]*m[2])**0.5*0.95]

       
        # weights=40#random.gauss(30.,3.)
        for h in range(num_repeats):
            x_int=[0.,0.,0.,0.,0.,0.]#[random.gauss(0.,0.025),0.]#[random.gauss(0.,0.1),random.gauss(0.,0.25)]#,random.gauss(0.,0.1),random.gauss(0.,0.25),random.gauss(0.,0.1),random.gauss(0.,0.25)]
            x0=[x_int[0]]
            x1=[x_int[1]]
            x2=[x_int[2]]
            x3=[x_int[3]]
            x4=[x_int[4]]
            x5=[x_int[5]]
            xdes=0.15#abs(random.gauss(0.15,0.01))  
            kp = random.gauss(20.,2.) 
            ki = 0.0#random.gauss(0.01,0.001) 
            # x_int=[0.,random.gauss(0.,0.25),0.,random.gauss(0.,0.25),0.,random.gauss(0.,0.25)]
            Fi=0
            F_0=[0]
            Fi_rec=0
            for i in range(0,len(t),int_t):
                
                tin=t[i:i+int_t+1]
                Fi = (xdes-x4[-1])*kp+Fi_rec*ki
                if random.randint(0,1)==1:
                    Fi=random.gauss(0.,0.5)
                Fi_rec+=Fi
                xd_prev.append(xdes)
                F_0.append(Fi)
                x = odeint(mydiff, x_int, tin,args=(m,k,c,Fi))
                # Fi=-0.01*(weights*(x[0,-1]-xdes))+Fi

                count=count*-1
                x0=np.append(x0,x[:-1,0])
                x1=np.append(x1,x[:-1,1])
                x2=np.append(x2,x[:-1,2])
                x3=np.append(x3,x[:-1,3])
                x4=np.append(x4,x[:-1,4])
                x5=np.append(x5,x[:-1,5])

                x_int=x[-1,:]

        # x0_out.append(normalize(x0))
        # x1_out.append(normalize(x1))
        x0_out.append(x0)
        x1_out.append(x1)
        x2_out.append(x2)
        x3_out.append(x3) 
        x4_out.append(x4)
        x5_out.append(x5)
        F_out.append(F_0)                  

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


    ## Save ground truth data to text file ##
    # out_data=1
    # zout=np.zeros((len(x0),2))
    # for i in range(len(x0)):
    #     zout[i,:]=np.array([x0[i],x1[i]])#,x2[i],x3[i],x4[i],x5[i]])

    # np.savetxt('pos'+str(out_data)+'.txt', zout, delimiter='\t', newline='\n')
    # a0=0
    # a1=0
    # a2=0
    # for i in range(len(x0)-1):
        # a0=np.append(a0,(x1[i+1]-x1[i])/(dt))
        # a1=np.append(a1,(x3[i+1]-x3[i])/(dt))
        # a2=np.append(a2,(x5[i+1]-x5[i])/(dt))

    # plt.plot(x0)
    # plt.show()
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import os

    ## Save as a datalist ##
    data=[]
    edge=torch.tensor
    count=0
    for j in range(len(x0_out)):
        for i in range(len(x0_out[0])-1):
            x=torch.tensor([x0_out[j][i],x1_out[j][i],x4_out[j][i],x5_out[j][i],F_out[j][i]],dtype=torch.float)
            y=torch.tensor([x0_out[j][i+1],x1_out[j][i+1],x4_out[j][i+1],x5_out[j][i+1],F_out[j][i+1]],dtype=torch.float)             
            # x=torch.tensor([x0_out[j][i],x1_out[j][i],x2_out[j][i],x3_out[j][i],x4_out[j][i],x5_out[j][i],F_out[j][i]],dtype=torch.float)
            # y=torch.tensor([x0_out[j][i+1],x1_out[j][i+1],x2_out[j][i+1],x3_out[j][i+1],x4_out[j][i+1],x5_out[j][i+1],F_out[j][i+1]],dtype=torch.float)  
            data.append([x,y])

    torch.save(data,os.path.join('./',f'data_{out_data}.pt'))
    return data, x0_out, x1_out

# run_sim(run_nums=1)