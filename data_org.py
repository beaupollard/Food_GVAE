import numpy as np
import torch
import os
import random
import math
xrec1=np.load('./data/xrec_rev8.npy')
urec1=np.load('./data/urec_rev8.npy')
urec1=np.clip(urec1,-20.,20.)
save_inds=[]
for i in range(len(xrec1)):
    if abs(math.pi-xrec1[i,-1,1])*180/math.pi<10:
        save_inds.append(i)
    # else:
    #     save_inds.append(i)

xrec=xrec1[save_inds,:,:]
urec=urec1[save_inds,:]
# xrec2=np.load('./data/xrec_rev4.npy')
# urec2=np.load('./data/urec_rev4.npy')
# xrec=np.concatenate((xrec1,xrec2))
# urec=np.concatenate((urec1,urec2))
data=[]
data_val=[]
index_rec=[1,3,4,5]
rand = np.array([random.randint(0, len(xrec)) for iter in range(int(len(xrec)*.1))])
rand=np.sort(rand)
for i in range(len(rand)-1):
    if rand[i]==rand[i+1]:
        rand[i+1]+=1
# rand=[0,10,15,30,45,60,75,85,93,100,110,119,126,135,145,156]
count=0
for i in range(len(xrec[:,0,0])):
    for j in range(len(xrec[0,:,0])-4):
        x=torch.tensor(np.concatenate((xrec[i,j,index_rec],np.array([urec[i,j]]))),dtype=torch.float)
        y=torch.tensor(np.concatenate((xrec[i,j+1,index_rec],np.array([urec[i,j+1]]))),dtype=torch.float)
        y1=torch.tensor(np.concatenate((xrec[i,j+2,index_rec],np.array([urec[i,j+2]]))),dtype=torch.float)
        y2=torch.tensor(np.concatenate((xrec[i,j+3,index_rec],np.array([urec[i,j+3]]))),dtype=torch.float)
        y3=torch.tensor(np.concatenate((xrec[i,j+4,index_rec],np.array([urec[i,j+4]]))),dtype=torch.float)
        # x=torch.tensor(np.concatenate((xrec[i,j,:],np.array([urec[i,j]]))),dtype=torch.float)
        # y=torch.tensor(np.concatenate((xrec[i,j+1,:],np.array([urec[i,j+1]]))),dtype=torch.float)
        # y1=torch.tensor(np.concatenate((xrec[i,j+2,:],np.array([urec[i,j+2]]))),dtype=torch.float)
        # y2=torch.tensor(np.concatenate((xrec[i,j+3,:],np.array([urec[i,j+3]]))),dtype=torch.float)
        # y3=torch.tensor(np.concatenate((xrec[i,j+4,:],np.array([urec[i,j+4]]))),dtype=torch.float)
        if i==rand[count]:
            data_val.append([x,y,y1,y2,y3])
        else:
            data.append([x,y,y1,y2,y3])
    if i==rand[count] and count<len(rand)-1:
        count+=1

torch.save(data,os.path.join('./',f'data_0808.pt'))        
torch.save(data_val,os.path.join('./',f'data_0808_val.pt'))     
