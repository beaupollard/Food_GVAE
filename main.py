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

BS=2048*8    # Batch size for training

## Load previously generated simulation data ##
d1=torch.load('./data_swing.pt')
d2=torch.load('./data_swing_val.pt')

## Setup data loader ##
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)
test=torch.utils.data.DataLoader(d2,batch_size=len(d2), shuffle=False)

## Initialize the NN model ##
model=VAE(enc_out_dim=len(d1[0][0])-1,input_height=len(d1[0][0])-1)

## Save model to either cpu or gpu ##
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)

# model.load_state_dict(torch.load("./models/swing_up")) 
count=0
model.human_rollout(test,device)
## Training loop ##
for i in range(10000):
    loss=model.training_human(train,device)
    if count==100:     
        model.scheduler.step()
        count=0
    count+=1
    print(i, loss)

torch.save(model.state_dict(), './models/swing_up')    # Save the current model


## Testing loop ##
model=VAE()

xhat, z, x, ztilde, zdecode = model.test(test,device)

sim_length=424
## Plot the latent space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(z[i:i+sim_length,0],z[i:i+sim_length,1])
plt.show()

## Plot the state space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(x[i:i+sim_length,0],x[i:i+sim_length,1])
plt.show()