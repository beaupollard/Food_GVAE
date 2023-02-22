import time
from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy
import numpy as np
import json
import matplotlib.pyplot as plt

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

def record_info():
    # return [sim.getObjectPosition(joint_ids[1],-1)[0], sim.getObjectPosition(joint_ids[1],-1)[2], sim.getJointPosition(joint_ids[0])]
    return [sim.getObjectPosition(joint_ids[1],-1)[0], sim.getObjectPosition(joint_ids[2],-1)[0], sim.getObjectPosition(joint_ids[3],-1)[0], sim.getObjectPosition(joint_ids[3],-1)[2], sim.getJointPosition(joint_ids[-1]), sim.getJointVelocity(joint_ids[0]), sim.getJointForce(joint_ids[0])]

stopper=False
joint_names=['/Input_joint','/Input_body','/Output','/Sphere','/Pen']#['/Pen','/Sphere']#
client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

joint_ids=get_ids(joint_names)
exp_data=read_inputs()
for i in range(4):
    exp_data[:,i]=(exp_data[:,i]-exp_data[0,i])/-1000.
# sim.setJointPosition(joint_ids[0],33.7348*math.pi/180.)
sim.startSimulation()
sim_data=[]
for i in exp_data[:,0]:
    sim.setJointTargetPosition(joint_ids[0],i)
    sim_data.append(record_info())
    client.step()

sim_data=np.array(sim_data)

for i in range(len(sim_data)):
    sim_data[:,i]=(sim_data[:,i]-sim_data[0,i])
plt.plot(exp_data[:,0])
plt.plot(sim_data[:,0])
# plt.plot(sim_data[:,2]*180/math.pi)
# plt.plot(exp_data[:,0])
plt.show()
print('stop')
