from optimal_ctrl import opt_ctrl
# from cartpole_ctrl import cart_pole
import numpy as np
from mujoco_runner import sim_runner
import math
import copy
import mujoco as mj
import matplotlib.pyplot as plt
import re
import random

def set_up_mj(filename):
    ## Setup Mujoco node ##
    active_names=['cart1']
    body_names=[{"name": 'cart1',"type": "joint", "qpos": [0], "qvel": [0]}, \
                {"name": 'pend',"type": "joint", "qpos": [0], "qvel": [0]}, \
                {"name": 'cart2',"type": "joint", "qpos": [0], "qvel": [0]}]
    sim=sim_runner(filename=filename,active_names=active_names,tracked_bodies=body_names,render=False)
    return sim

def change_line(target,new):
    for i in range(len(lines)):
        lines[i] = re.sub(target,new,lines[i])

    f = open(filename, "w+")
    f.writelines(lines)
    f.close()
    return new

def random_config():
    global target
    rand=random.randint(0, 2)
    k = 15
    if rand==0:
        # underdamped #
        beta = random.uniform(0.5, 0.95)
        cc = 2*mass*(k/mass)**0.5
        b = beta*cc
    if rand==1:
        b = 2*mass*(k/mass)**0.5
    else:
        # overdamped #
        beta = random.uniform(1.05, 3.0)
        cc = 2*mass*(k/mass)**0.5
        b = beta*cc
    new = 'stiffness="'+"{:4.1f}".format(k)+'" damping="'+"{:8.4f}".format(b)+'"'
    change_line(target,new)
    target = copy.copy(new)

target = 'stiffness="200.0" damping="10.0"'

## Cart Params ##
mass = 0.537

filename='assets/mass_cart_pend2.xml'

with open('assets/mass_cart_pend.xml') as f:
    lines = f.readlines()

sim=set_up_mj(filename='assets/mass_cart_pend.xml')

Q=np.zeros((sim.nstate,sim.nstate))#0.01*np.eye((sim.nstate))
Q[1,1]=1.0
Q[0,0]=0.01
Q[-2,-2]=0.1
# Q[-1,-1]=0.01

Q_term=copy.copy(Q)
Q_term[0,0]=20.

Q_term[1,1]=1000.
Q_term[-2,-2]=100.

R=0.01*np.eye(sim.nact)
tspan=500
xdes=sim.get_state(sim.data)
xdes[1]=math.pi
u_init=np.zeros((tspan,sim.nact))
for i in range(tspan):
    u_init[i]=-5.0*math.sin(2*math.pi*i*0.01)
num_runs=300 
# for i in range(999):
#     # sim.data.ctrl[0]=u_init[i]
#     sim.step_forward([u_init[0,i]])
    # sim.render_image()
urec=np.zeros((num_runs,tspan*2))
xrec=np.zeros((num_runs,tspan*2,len(xdes)))
rec_count=0
for i in range(num_runs):
    rec_info=True
    random_config()    
    sim.random_configuration()
    sim=set_up_mj(filename)
    ctrl = opt_ctrl(sim,xdes=xdes,tspan=tspan,Q=Q,R=R,Q_term=Q_term,u_init=u_init)
    Uout, xout = ctrl.ilqr()
    us=[]
    xs=[]
    for j in range(tspan*2):
        
        if j<len(Uout):
            u0=Uout[j]
        else:
            if abs(sim.get_state(sim.data)[1]-xdes[1])>45*math.pi/180:
                rec_info=False
            u0=ctrl.lqr()
        us.append(u0)
        sim.step_forward(u0)
        xs.append(sim.get_state(sim.data))
    if rec_info:
        urec[rec_count,:]=np.array(us).flatten()
        xrec[rec_count,:,:]=np.array(xs)
        rec_count+=1
np.save('xrec_rev2',xrec[:rec_count,:,:])
np.save('urec_rev2',urec[:rec_count,:])

# for i in range(tspan):
#     sim.data.ctrl[0]=u_init[i]
#     mj.mj_step(sim.model,sim.data)
#     sim.render_image()
# sim.data.qpos[2]=math.pi+math.pi/180*6.
# ctrl = opt_ctrl(sim,xdes=xdes,tspan=tspan,Q=Q,R=R,Q_term=Q_term,u_init=u_init)
# for i in range(500):
#     u0=ctrl.lqr()
#     sim.step_forward(u0)
#     sim.render_image()


# labels=['m0','theta','m1','v0','omega','v1']
# fig, axs = plt.subplots(3)
# fig.suptitle('State Space')         
# u_init=np.load('urec.npy')
# xrec=np.load('xrec.npy')
# t=np.linspace(0,10,len(xrec[0,:,0]))    
# for i in range(30):
#     for j in range(len(xrec[0,0,:3])):
#         axs[j].plot(t,xrec[i,:,j],linewidth=4,color='b')

#         axs.flat[j].set(xlabel='time (s)',ylabel=labels[j])

# plt.savefig('States.png',bbox_inches='tight')    