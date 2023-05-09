from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy
import time
import torch



BS=2048*4    # Batch size for training

## Load previously generated simulation data ##
sim=torch.load('./data/data_sim_pos.pt')

## Setup data loader ##
train=torch.utils.data.DataLoader(sim,batch_size=len(sim), shuffle=False)

## Initialize the NN model ##
model=VAE(enc_out_dim=len(sim[0][0])-1,input_height=len(sim[0][0])-1)

## Save model to either cpu or gpu ##
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)

## Load the model that was trained on the human data ##
model.load_state_dict(torch.load("./models/human_LTI"))     

count=0
## Training loop ##
for i in range(10000):
    loss=model.training_sim(train,device)
    if count==500:
        model.scheduler.step() # decrease the learning rate
        count=0
    count+=1
    print(i, loss)

torch.save(model.state_dict(), './models/robot_LTI')    # Save the current model


## Testing loop ##
test_sim=torch.utils.data.DataLoader(sim,batch_size=len(sim), shuffle=False)

xhat_sim, z_sim, x_sim, z_sim_tilde, z_sim_1, u = model.test_sim(test_sim,device)

# sim_length=424
# ## Plot the latent space phase portrait ##
# for i in range(0,len(x),sim_length):
#     plt.plot(z[i:i+sim_length,0],z[i:i+sim_length,1])
# plt.show()

# ## Plot the state space phase portrait ##
# for i in range(0,len(x),sim_length):
#     plt.plot(x[i:i+sim_length,0],x[i:i+sim_length,1])
# plt.show()