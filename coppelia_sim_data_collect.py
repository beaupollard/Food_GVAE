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

def record_info():
    # return [sim.getObjectPosition(joint_ids[1],-1)[0], sim.getObjectPosition(joint_ids[1],-1)[2], sim.getJointPosition(joint_ids[0])]
    return [sim.getJointForce(joint_ids[0]), sim.getObjectPosition(joint_ids[1],-1)[0], sim.getObjectPosition(joint_ids[2],-1)[0], sim.getObjectPosition(joint_ids[3],-1)[0], sim.getObjectPosition(joint_ids[3],-1)[2]]

def reset_scene():
    # sim.setJointPosition(joint_ids[-1],0.)
    # sim.setJointPosition(joint_ids[-1],0.)
    pos0 = abs(sim.getObjectPosition(joint_ids[1],-1)[0]+exp_data[0,0])
    j0 = abs(sim.getJointPosition(joint_ids[-1]))
    dj0dt = abs(sim.getJointVelocity(joint_ids[-1]))
    if pos0<0.015:
        sim.setJointPosition(joint_ids[-1],0.)
        sim.resetDynamicObject(joint_ids[3])
        sim.resetDynamicObject(joint_ids[5])
        sim.resetDynamicObject(joint_ids[1])
        sim.resetDynamicObject(joint_ids[2])
        sim.resetDynamicObject(joint_ids[6])
        return True
    else:
        return False
    

stopper=False
joint_names=['/Motor_input','/Input_body','/Output','/Sphere','/Link','/l0','/l1','/Pen']#['/Pen','/Sphere']#
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.startSimulation()
# sim.closeScene() 
client.setStepping(True)


exp_data=read_inputs()
for i in range(4):
    if i==3:
        sign_multi=1
    else:
        sign_multi=-1
    exp_data[:,i]=sign_multi*(exp_data[:,i]-exp_data[0,i])/1000.


sim_data=[]
for j in range(20):
    joint_ids=get_ids(joint_names)
    kp=np.random.normal(60.,5.)
    for i in exp_data[:,0]:
        ctrl(i,kp)
        sim_data.append(record_info())
        client.step()

    move_on=False
    while move_on==False:
        ctrl(exp_data[1,0],15.,0.,0.)
        move_on=reset_scene()
        client.step()
        

sim_data=np.array(sim_data)

for i in range(4):
    sim_data[:,i+1]=(sim_data[:,i+1]-sim_data[0,i+1])

## Save as a datalist ##
data=[]
edge=torch.tensor
count=0
for j in range(len(sim_data[:,0])-1):
    x=torch.tensor([sim_data[j,1],sim_data[j,2],sim_data[j,3],sim_data[j,4],sim_data[j,0]],dtype=torch.float)
    y=torch.tensor([sim_data[j+1,1],sim_data[j+1,2],sim_data[j+1,3],sim_data[j+1,4],sim_data[j+1,0]],dtype=torch.float)
    data.append([x,y])

torch.save(data,os.path.join('./',f'data_sim.pt'))
# return data, len(t)-1, x0_out, x1_out

# plt.plot(exp_data[:,0])
# plt.plot(-sim_data[:,0])
# plt.plot(sim_data[:,2]*180/math.pi)
# plt.plot(exp_data[:,0])
# plt.show()
# print('stop')
