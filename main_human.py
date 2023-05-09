from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy
import time
import torch
from scipy import signal

BS=2048*8    # Batch size for training

## Load previously generated simulation data ##
d1=torch.load('./data/data_exp_pos.pt')

## Setup data loader ##
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

## Initialize the NN model ##
model=VAE(enc_out_dim=len(d1[0][0]),input_height=len(d1[0][0]))

## Save model to either cpu or gpu ##
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)

count=0

## Training loop ##
for i in range(10000):
    loss=model.training_human(train,device) 
    if count==100:     
        model.scheduler.step() # Decrease the learning rate
        count=0
    count+=1
    print(i, loss)

torch.save(model.state_dict(), './models/human_LTI')    # Save the current model


## Testing loop ##
test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
xhat, z, x, ztilde, zdecode = model.test_human(test,device)

sim_length=424
## Plot the latent space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(z[i:i+sim_length,0],z[i:i+sim_length,1])
plt.show()

## Plot the state space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(x[i:i+sim_length,0],x[i:i+sim_length,1])
plt.show()