from optimal_ctrl import opt_ctrl
# from cartpole_ctrl import cart_pole
import numpy as np
from mujoco_runner import sim_runner
import math
import copy
import mujoco as mj
import matplotlib.pyplot as plt

## Setup Mujoco node ##
filename='assets/mass_cart_pend.xml'
active_names=['cart1']
body_names=[{"name": 'cart1',"type": "joint", "qpos": [0], "qvel": [0]}, \
            {"name": 'pend',"type": "joint", "qpos": [0], "qvel": [0]}, \
            {"name": 'cart2',"type": "joint", "qpos": [0], "qvel": [0]}]
sim=sim_runner(filename=filename,active_names=active_names,tracked_bodies=body_names,render=False)

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
    u_init[i]=5.0*math.sin(2*math.pi*i*0.01)
    
# for i in range(999):
#     # sim.data.ctrl[0]=u_init[i]
#     sim.step_forward([u_init[0,i]])
    # sim.render_image()
urec=np.zeros((30,tspan*2))
xrec=np.zeros((30,tspan*2,len(xdes)))
rec_count=0
for i in range(30):
    rec_info=True
    sim.random_configuration()
    ctrl = opt_ctrl(sim,xdes=xdes,tspan=tspan,Q=Q,R=R,Q_term=Q_term,u_init=u_init)
    Uout, xout = ctrl.ilqr()
    us=[]
    xs=[]
    for j in range(tspan*2):
        
        if j<len(Uout):
            u0=Uout[j]
        else:
            if abs(sim.get_state(sim.data)[1]-math.pi)>45*math.pi/180:
                rec_info=False
            u0=ctrl.lqr()
        us.append(u0)
        sim.step_forward(u0)
        xs.append(sim.get_state(sim.data))
    if rec_info:
        urec[rec_count,:]=np.array(us).flatten()
        xrec[rec_count,:,:]=np.array(xs)
        rec_count+=1
np.save('xrec',xrec)
np.save('urec',urec)

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