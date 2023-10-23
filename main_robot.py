from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import springmassdamper as smd
import copy
import time
import torch
from scipy import signal
from latent_LQR_func import run_test

BS=int(5*512)    # Batch size for training

## Load previously generated simulation data ##
# d1=torch.load('./data_0829_over.pt')
# d2=torch.load('./data_0829_over_val.pt')
d1=torch.load('./data_single_0913v2.pt')
d2=torch.load('./data_single_0913_valv2.pt')
d3=torch.load('./human_0912.pt')
# d2=torch.load('./human_0912_val.pt')
# d3=torch.load('./data_0911.pt')

## Setup data loader ##
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)
test=torch.utils.data.DataLoader(d2,batch_size=len(d2), shuffle=False)
# test2=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)

## Initialize the NN model ##
model=VAE(enc_out_dim=len(d1[0][0])-1,input_height=len(d1[0][0])-1)

## Save model to either cpu or gpu ##
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")    #Save the model to the CPU
model.to(device)
model.load_state_dict(torch.load('./models/human_test2v2')) 
# model.load_state_dict(torch.load('./models/model_0912v2')) 
# model.load_state_dict(torch.load('./models/model_0823_varied')) 
# model.load_state_dict(torch.load('./models/model_0816_pend_pos')) 
# model.load_state_dict(torch.load('./models/model_3D')) 
count=0

# model.human_rollout(test,device)
loss_test_rec=[]
loss_train_rec=[]
klf_lr=5.0
kl_init_lr=0.5
gamma=0.05
rec_info=[]
## Training loop ##
for i in range(1000):
    loss=model.training_sim(train,device)
    _, _, _, _, loss_test = model.test_sim(test,device)
    loss_train_rec.append(sum(loss))
    loss_test_rec.append(sum(loss_test))
    if count==100:     
        model.scheduler.step()
        count=0
    theta,dtheta=run_test(model)
    rec_info.append([theta,dtheta])
    if abs(theta-math.pi)<0.1 and abs(dtheta)<0.1:
        torch.save(model.state_dict(), './models/robot_0915_success_nohuman')
        break
    # model.kl_weight=(klf_lr+kl_init_lr)-klf_lr*math.exp(-gamma*i/50)

    count+=1
    print(i, loss)

torch.save(model.state_dict(), './models/robot_0912v5')    # Save the current model

zout,_=model.human_rollout(test,device)
test2=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
xhat, z, x, ztilde, zdecode, loss = model.test_human(test2,device)
for ii in range(int(len(z)/996)):
    plt.plot(z[ii*996:(ii+1)*996,0],z[ii*996:(ii+1)*996,1],'b')


zout2=model.sim_rollout(device,iters=990,z_init=zout[0])
plt.plot(zout2[:,0],zout2[:,1],'r')


zout3,_=model.sim_rob_rollout(device,iters=900,z_init=zout[0])

plt.plot(zout3[:,0],zout3[:,1],'g')


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for ii in range(int(len(z)/996)):
#     ax.plot3D(z[ii*996:(ii+1)*996,0],z[ii*996:(ii+1)*996,1],z[ii*996:(ii+1)*996,2],'b')
# ax.plot3D(zout2[:,0],zout2[:,1],zout2[:,2],'r')
# ax.plot3D(zout3[:,0],zout3[:,1],zout3[:,2],'g')

## Testing loop ##
model=VAE()

xhat, z, x, ztilde, zdecode, loss = model.test_human(test,device)
for ii in range(6):
    plt.subplot(2, 3, ii+1)
    plt.plot(x[:,ii])
    plt.plot(xhat[:,ii])
plt.show()
sim_length=424
## Plot the latent space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(z[i:i+sim_length,0],z[i:i+sim_length,1])
plt.show()

## Plot the state space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(x[i:i+sim_length,0],x[i:i+sim_length,1])
plt.show()

# for i in range(144):
#     for ii in range(3):
#         plt.subplot(1, 3, ii+1)
#         plt.plot(z2[i*996:(i+1)*996,ii],'b')
# for i in range(20):
#     for ii in range(3):
#         plt.subplot(1, 3, ii+1)
#         plt.plot(z[i*996:(i+1)*996,ii],'b')
# for ii in range(3):
#     plt.subplot(1, 3, ii+1)
#     plt.plot(zzout[:,ii],'r')