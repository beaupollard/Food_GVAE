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
    plt.plot(output[:1000],output2[:1000])
    plt.plot(output[-1000:],output2[-1000:])
    plt.show()

num_repeats=10
run_nums=4

BS=128#256
percent_train=0.8
d1=smd.run_sim(run_nums=30,out_data=2,num_repeats=1)

train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

model=VAE()
# model.load_state_dict(torch.load("./current_model2"))
for i in range(400):
    loss=model.training_step(train)
    if i==200:
        plot_latent_smooth()
    print(i, loss)
torch.save(model.state_dict(), 'current_model6')


# ## Test ##
model=VAE()

model.load_state_dict(torch.load("./current_model5"))
model.eval()
d2=torch.load('data_1.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
test=torch.utils.data.DataLoader(d2,batch_size=len(d2), shuffle=False)
xhat, z, x = model.test(test)
# animation_test.animate_latent(z,'latentunder.mp4','b',0,1000,'z')
# animation_test.animate_latent(z,'latentcritical.mp4','r',8991,8991+1000,'z')
# animation_test.animate_latent(z,'latentover.mp4','k',21978,21978+1000,'z')
# animation_test.animate_latent(x,'xunder.mp4','b',0,1000,'x')
# animation_test.animate_latent(x,'xcritical.mp4','r',8991,8991+1000,'x')
# animation_test.animate_latent(x,'xover.mp4','k',21978,21978+1000,'x')
# animation_test.animate_pos(x[:1000,0],0*x[:1000,0],'under_damped.mp4','b')
# animation_test.animate_pos(x[8991:8991+1000,0],0*x[8991:8991+1000,0],'critical_damped.mp4','b')
# animation_test.animate_pos(x[21978:21978+1000,0],0*x[21978:21978+1000,0],'over_damped.mp4','k')
# for i in range(10000,20000,1000):
#     plt.plot(z[i:i+400,0],z[i:i+400,1],'b')
# x_coll=np.reshape(x[:,0],(499,4))
# v_coll=np.reshape(x[:,1],(499,4))

# plt.plot(x[:499,0],x[:499,1])
# plt.plot(x[499:499*2,0],x[499:499*2,1])
# plt.plot(x[499*2:499*3,0],x[499*2:499*3,1])
# print('hey')


