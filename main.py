from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import springmassdamper as smd
import copy
import time
import torch

num_repeats=10
run_nums=4

BS=512
percent_train=0.8
d1=smd.run_sim(run_nums=30,out_data=1,num_repeats=1)

# latent_multi=1.

# datalist=random.sample(d1,len(d1))
# length_d=int(percent_train*len(datalist))
# data_train2=datalist[0:length_d]

# data_train=torch.utils.data.DataLoader(data_train2[:math.floor(length_d/BS)*BS],batch_size=BS,shuffle=True)

# BS2=len(datalist[length_d:])
# data_test=torch.utils.data.DataLoader(datalist[length_d:],batch_size=BS2)
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

model=VAE()
model.load_state_dict(torch.load("./current_model2"))
for i in range(10000):
    loss=model.training_step(train)
    print(i, loss)
torch.save(model.state_dict(), 'current_model2')
# d1=torch.load('data_1.pt')
# run_nums=4
# d1=smd.run_sim(run_nums=run_nums,out_data=2,num_repeats=1)


# ## Test ##
model=VAE()

model.load_state_dict(torch.load("./current_model"))
model.eval()
d2=smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
test=torch.utils.data.DataLoader(d2,batch_size=len(d2), shuffle=False)
xhat, z, x = model.test(test)
for i in range(400,len(x),500):
    plt.plot(x[i:i+99,0],x[i:i+99,1])
# x_coll=np.reshape(x[:,0],(499,4))
# v_coll=np.reshape(x[:,1],(499,4))

# plt.plot(x[:499,0],x[:499,1])
# plt.plot(x[499:499*2,0],x[499:499*2,1])
# plt.plot(x[499*2:499*3,0],x[499*2:499*3,1])
# print('hey')