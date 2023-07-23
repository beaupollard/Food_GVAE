import time
from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import os
from VAE import VAE

def read_inputs():
    lines_out=[]
    # with open('./pend.txt') as f:
    with open('./exp_res.txt') as f:
        lines = f.readlines()   
    data_out=[]
    for j in lines:
        str_list=j.replace('\n','').split('\t')
        float_list=[float(x) for x in str_list]
        data_out.append(float_list)
    return np.array(data_out)

def get_ids(names):
    joint_ids=[]
    for i in names:
        joint_ids.append(sim.getObject(i))
    return joint_ids

def ctrl(xdes,kp=60.,mu=0.,std=0.1):
    
    ctrl_inp=kp*(sim.getObjectPosition(joint_ids[1],-1)[0]-xdes)+np.random.normal(mu,std)
    sim.setJointTargetForce(joint_ids[0],ctrl_inp)

def ctrl_force(ctrl_inp):
    
    sim.setJointTargetForce(joint_ids[0],ctrl_inp)

def record_info():
    # return [sim.getObjectPosition(joint_ids[1],-1)[0], sim.getObjectPosition(joint_ids[1],-1)[2], sim.getJointPosition(joint_ids[0])]
    return [sim.getObjectPosition(joint_ids[1],-1)[0], sim.getObjectPosition(joint_ids[2],-1)[0], sim.getObjectPosition(joint_ids[3],-1)[0], sim.getObjectPosition(joint_ids[3],-1)[2], sim.getJointPosition(joint_ids[-1]), sim.getJointVelocity(joint_ids[0]), sim.getJointForce(joint_ids[0])]


stopper=False
joint_names=['/Input_joint','/Input_body','/Output','/Sphere','/Spherev0','/Link','/Pen']
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.startSimulation()
# sim.closeScene() 
client.setStepping(True)

joint_ids=get_ids(joint_names)

d1=torch.load('./data_swing.pt')
test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)

model=VAE(enc_out_dim=6,input_height=6)
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)
model.load_state_dict(torch.load("./models/swing_up_ctrl"))     # Load a previously trained model
xhat, z, x, zout, zt1, u = model.test_sim(test,device)

count=0
sim_data=[]
sim_data.append(record_info())
# _, _, ctrl_inp = model.project_forward(0*torch.tensor([sim_data[-1][:4]],dtype=torch.float),device)
for i in range(400):
    current_state=torch.tensor(record_info(),dtype=torch.float)
    # _, _, ctrl_inp = model.get_ctrl_from_human(d1[i],device)
    # ctrl_force(ctrl_inp[0].item())
    ctrl_inp = model.get_ctrl_LQR(z[i],current_state,device)
    ctrl_force(ctrl_inp.item())    
    sim_data.append(record_info())    
    client.step()
print("stop")