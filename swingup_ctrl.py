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
from latent_dynamics import latent_dynamics

## Setup Mujoco node ##
filename='assets/mass_cart_pend2.xml'
active_names=['cart1']
body_names=[{"name": 'cart1',"type": "joint", "qpos": [0], "qvel": [0]}, \
            {"name": 'pend',"type": "joint", "qpos": [0], "qvel": [0]}, \
            {"name": 'cart2',"type": "joint", "qpos": [0], "qvel": [0]}]
sim=sim_runner(filename=filename,active_names=active_names,tracked_bodies=body_names,render=False)
sim.random_configuration()

d1=torch.load('./data_0816_single_val.pt')
test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)

model=VAE(enc_out_dim=len(d1[0][0])-1,input_height=len(d1[0][0])-1)
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)
model.load_state_dict(torch.load('./models/robot_0822v3'))     # Load a previously trained model
# model.eval()
xhat, z, x, zout, zt1, u = model.test_human(test,device)
state_rec=[]
u_rec=[]
tspan=200
Q=10*np.eye(2)
Q_term=100*Q
R=1.0*np.eye(1)
# u_init=x[:tspan,-1]#-0*np.ones((tspan,1))

index_state=[1,3,4,5]
# sim.random_configuration()
current_state=torch.tensor(sim.get_state(sim.data),dtype=torch.float)[index_state]
latent_sim = latent_dynamics(state_dim=2,action_dim=1,current_state=current_state,model=model)
u_init=latent_sim.u_traj#-0*np.ones((tspan,1))
ctrl = opt_ctrl(latent_sim,xdes=latent_sim.z_traj[tspan-1,:],tspan=tspan,Q=Q,R=R,Q_term=Q_term,u_init=u_init)
xs=[]
## Get target ## Given an A and B 
for i in range(990):
    ctrl.xdes=latent_sim.z_traj[i+tspan,:]
    current_state=torch.tensor(sim.get_state(sim.data),dtype=torch.float)[index_state]
    latent_sim.set_current_state(current_state)
    state_rec.append(sim.get_state(sim.data))
    Uout, xout = ctrl.ilqr(max_iter=10)
    # plt.plot(latent_sim.z_traj[:,0],latent_sim.z_traj[:,1])
    # plt.plot(xout[:,0],xout[:,1])    
    ctrl.U=np.concatenate((Uout[1:],np.array([[1.]])))
    u_rec.append(copy.copy(Uout[0][0]))
    sim.step_forward(u=[Uout[0][0]])
    xs.append(sim.get_state(sim.data))
    # print("stop")
    # ctrl_inp = model.get_ctrl_LQR(z[i],current_state,device)
    # u_rec.append(copy.copy(ctrl_inp))
    # sim.step_forward(u=[ctrl_inp])