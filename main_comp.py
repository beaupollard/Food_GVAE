from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import springmassdamper as smd
import copy
import time
import torch
import animation_test
from scipy import signal

def plot_latent_smooth():
    test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
    xhat, z, x = model.test(test)
    fs=1/0.1
    fc = 1.  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, z[:,0])
    output2 = signal.filtfilt(b, a, z[:,1])
    for i in range(10000,20000,1000):
        plt.plot(output[i:i+400],output2[i:i+400],'b')
    plt.plot(output[:1000],output2[:1000],'r')
    plt.plot(output[-1000:],output2[-1000:],'y')
    plt.show()


BS=2048*4    # Batch size for training

## Run new simulations ##
# d1, sim_length, _, _=smd.run_multimass_sim(run_nums=30,out_data=3,num_repeats=1)  # run simulation of 3 masses and a pendulum
# d1, sim_length, _, _=smd.run_singlemass_sim(run_nums=30,out_data=3,num_repeats=1)   # run simulation of single mass system

## Load previously generated simulation data ##
# exp=torch.load('data_exp_osc_02142023.pt')
exp=torch.load('data_sim.pt')
for i in range(len(exp)):
    exp[i][0][:-1]=1000*exp[i][0][:-1]
    exp[i][1][:-1]=1000*exp[i][1][:-1]
    exp[i][0][2]=-exp[i][0][2]
    exp[i][1][2]=-exp[i][1][2]
train=torch.utils.data.DataLoader(exp,batch_size=BS, shuffle=False)

model=VAE(enc_out_dim=len(exp[0][0])-1,input_height=len(exp[0][0])-1)
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)
model.load_state_dict(torch.load("./current_model0"))     # Load a previously trained model
count=0


## Training loop ##
for i in range(10000):
    loss=model.training_sim(train,device)
    if count==10:
        model.scheduler.step()
        count=0
    count+=1
    print(i, loss)

torch.save(model.state_dict(), 'current_model8')    # Save the current model


## Testing loop ##
model=VAE()

test=torch.utils.data.DataLoader(exp,batch_size=len(exp), shuffle=False)
xhat, z, x = model.test(test,device)

sim_length=424
## Plot the latent space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(z[i:i+sim_length,0],z[i:i+sim_length,1])
plt.show()

## Plot the state space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(x[i:i+sim_length,0],x[i:i+sim_length,1])
plt.show()