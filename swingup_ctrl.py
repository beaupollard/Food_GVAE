from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy
import time
import torch
from optimal_ctrl import opt_ctrl
from mujoco_runner import sim_runner
import mujoco as mj

## Setup Mujoco node ##
filename='assets/mass_cart_pend.xml'
active_names=['cart1']
body_names=[{"name": 'cart1',"type": "joint", "qpos": [0], "qvel": [0]}, \
            {"name": 'pend',"type": "joint", "qpos": [0], "qvel": [0]}, \
            {"name": 'cart2',"type": "joint", "qpos": [0], "qvel": [0]}]
sim=sim_runner(filename=filename,active_names=active_names,tracked_bodies=body_names,render=True)


d1=torch.load('./data_swing_val.pt')
test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)

model=VAE(enc_out_dim=len(d1[0][0])-1,input_height=len(d1[0][0])-1)
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)
model.load_state_dict(torch.load("./models/swing_up3"))     # Load a previously trained model
xhat, z, x, zout, zt1, u = model.test_human(test,device)
state_rec=[]
u_rec=[]
## Get target ##
for i in range(990):
    current_state=torch.tensor(sim.get_state(sim.data),dtype=torch.float)
    state_rec.append(sim.get_state(sim.data))
    ctrl_inp = model.get_ctrl_LQR(z[i],current_state,device)
    u_rec.append(copy.copy(ctrl_inp.item()))
    sim.step_forward(u=[ctrl_inp.item()])