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

def plot_latent_smooth(xinp,yinp,zinp):
    fs=1/0.1
    fc = 1.  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, xinp)
    output2 = signal.filtfilt(b, a, yinp)
    output3 = signal.filtfilt(b, a, zinp)
    # plt.plot(output,output2)
    # for i in range(10000,20000,1000):
    #     plt.plot(output[i:i+400],output2[i:i+400],'b')
    # plt.plot(output[:1000],output2[:1000],'r')
    # plt.plot(output[-1000:],output2[-1000:],'y')
    # plt.show()
    return np.array([output,output2,output3])


BS=2048*8    # Batch size for training

## Run new simulations ##
# d1, sim_length, _, _=smd.run_multimass_sim(run_nums=30,out_data=3,num_repeats=1)  # run simulation of 3 masses and a pendulum
# d1, sim_length, _, _=smd.run_singlemass_sim(run_nums=30,out_data=3,num_repeats=1)   # run simulation of single mass system

## Load previously generated simulation data ##
d1=torch.load('./data/data_exp_pos.pt')
# d1=torch.load('./data_exp_osc_02142023.pt')
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

model=VAE(enc_out_dim=len(d1[0][0]),input_height=len(d1[0][0]))
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)
# model.load_state_dict(torch.load("./models/current_modelrobot"))     # Load a previously trained model
# model2=VAE(enc_out_dim=len(d1[0][0]),input_height=len(d1[0][0]))
# device2 = torch.device("cpu")    # Save the model to the CPU
# model2.to(device)
# model2.load_state_dict(torch.load("./models/current_model9"))
count=0
## Training loop ##
for i in range(4000):
    loss=model.training_step(train,device)
    if count==200:
        # model.kl_weight=1.
        # model.lin_weight=10.0        
        model.scheduler.step()
        count=0
    count+=1
    print(i, loss)

torch.save(model.state_dict(), './models/human_model_pos')    # Save the current model


## Testing loop ##
model=VAE()

test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
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