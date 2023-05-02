from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
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


BS=2048*4    # Batch size for training

## Run new simulations ##
# d1, sim_length, _, _=smd.run_multimass_sim(run_nums=30,out_data=3,num_repeats=1)  # run simulation of 3 masses and a pendulum
# d1, sim_length, _, _=smd.run_singlemass_sim(run_nums=30,out_data=3,num_repeats=1)   # run simulation of single mass system

## Load previously generated simulation data ##
# exp=torch.load('data_exp_osc_02142023.pt')
sim=torch.load('./data_swing.pt')

train=torch.utils.data.DataLoader(sim,batch_size=len(sim), shuffle=False)

model=VAE(enc_out_dim=len(sim[0][0])-1,input_height=len(sim[0][0])-1)
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)

model.load_state_dict(torch.load("./models/swing_up"))     # Load a previously trained model
count=0

test_sim=torch.utils.data.DataLoader(sim,batch_size=len(sim), shuffle=False)

xhat_sim, z_sim, x_sim, z_sim_tilde, z_sim_1, u = model.test_sim(test_sim,device)
## Training loop ##
for i in range(10000):
    loss=model.training_sim(train,device)
    if count==500:
        model.scheduler.step()
        count=0
    count+=1
    print(i, loss)

torch.save(model.state_dict(), './models/swing_up_ctrl')    # Save the current model


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