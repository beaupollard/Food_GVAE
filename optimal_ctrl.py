import numpy as np
from scipy.spatial.transform import Rotation as R
import control
import math
import copy
import scipy
import matplotlib.pyplot as plt

class opt_ctrl():
    def __init__(self,dyn_model,xdes=[],tspan=2000,Q=[],R=[],Q_term=[],u_init=[],render=False):
        self.dyn_model=dyn_model
        self.tspan=tspan
        self.xdes=xdes
        self.R=R
        self.Q=Q
        self.Q_term=Q_term
        self.state_size=dyn_model.nstate
        self.nact=dyn_model.nact
        self.X=np.empty((tspan,dyn_model.nstate))
        self.mu_min=1e-6
        if len(u_init)==0:
            self.U=-5*np.ones((tspan,self.nact))
        else:
            self.U=u_init

    def lqr(self):
        A, B = self.dyn_model.forward_difference(eps=1e-6)
        
        # P=scipy.linalg.solve_continuous_are(A,B,self.Q,self.R)
        # K = np.linalg.pinv(B.T@P@B+self.R)@(B.T@P@A)
        K, _, _ = control.dlqr(A,B,self.Q,self.R)
        u=-K@(self.dyn_model.get_state(self.dyn_model.data)-self.xdes)

        return u

    def ilqr(self, max_iter=1000):
        self.mu = 1.0
        self.delta0 = 2.0
        self.delta=copy.copy(self.delta0)
        alphas = 1.1**(-np.arange(10)**2)
        dyn_model = self.dyn_model
        update = True
        Jopt=1e10
        conv=100.
        for iteration in range(max_iter):
            if update:

                xs, fx, fu, l_prev, lx, lxx, lu, luu, lux = self.forward_rollout(dyn_model,self.U)
                dyn_model.reset_data()
                # Jopt = sum(l_prev)
                update = False
                
            k, K = self.backward_pass(fx, fu, lx, lxx, lu, luu, lux)
            accepted=False
            for alpha in alphas:
                xs_new, u_new, l = self.new_ctrl(dyn_model,xs,self.U,k,K,alpha)
                dyn_model.reset_data()
                if sum(l) <= Jopt:
                    xs = copy.copy(xs_new)
                    self.U = copy.copy(u_new)
                    self.delta = min(1.0,self.delta)/self.delta0
                    self.mu *= self.delta
                    if self.mu <= self.mu_min:
                        self.mu=0.
                    update = True
                    conv=abs((sum(l)-Jopt)/Jopt*100.)
                    Jopt=copy.copy(sum(l))
                    accepted=True
                    break

            if not accepted:
                self.delta=max(1.0,self.delta)*self.delta0
                self.mu = max(self.mu_min,self.mu*self.delta)
                if self.mu>1e10:
                    print("regularization term to big")
                    self.mu=1.
                    break
            print(iteration, accepted, Jopt)
            if conv<0.1:
                return self.U, xs
        return self.U, xs

    def forward_rollout(self,dyn_model,u):
        xs = np.empty_like(self.X)
        fx = np.empty((self.tspan-1,self.state_size,self.state_size))
        fu = np.empty((self.tspan-1,self.state_size,self.nact))
        l = np.empty(self.tspan) 
        lx = np.empty((self.tspan,self.state_size))
        lu = np.empty((self.tspan-1,self.nact))
        lux = np.empty((self.tspan-1,self.nact,self.state_size))
        lxx = np.empty((self.tspan,self.state_size,self.state_size))
        luu = np.empty((self.tspan-1,self.nact,self.nact))
        xs[0,:]=dyn_model.get_state(dyn_model.data)

        for i in range(self.tspan-1):
            ## step simulation forward ##
            self.dyn_model.step_forward(u[i])
            xs[i+1,:]=dyn_model.get_state(dyn_model.data)

            A, B = self.dyn_model.forward_difference(eps=1e-6)
            fx[i] = A#np.eye(self.state_size)+A*dyn_model.dt
            fu[i] = B#*dyn_model.dt           
            l[i], lx[i], lxx[i], lu[i], luu[i], lux[i] = self.costs(xs[i],u[i],terminal=False)

        l[-1], lx[-1], lxx[-1] = self.costs(xs[-1],u[-1],terminal=True)

        return xs, fx, fu, l, lx, lxx, lu, luu, lux
    
    def backward_pass(self,fx, fu, lx, lxx, lu, luu, lux):
        Vx = lx[-1]
        Vxx = lxx[-1]

        k = np.empty((self.tspan-1,self.nact))
        K = np.empty((self.tspan-1,self.nact,self.state_size))

        for i in range(self.tspan-2,-1,-1):
            Qx = lx[i]+fx[i].T.dot(Vx)
            Qu = lu[i]+fu[i].T.dot(Vx)
            Qxx = lxx[i]+fx[i].T.dot(Vxx).dot(fx[i])

            reg = self.mu*np.eye(self.state_size)
            Qux = lux[i] + fu[i].T.dot(Vxx + reg).dot(fx[i])
            Quu = luu[i] + fu[i].T.dot(Vxx + reg).dot(fu[i])

            k[i] = -np.linalg.solve(Quu,Qu)
            K[i] = -np.linalg.solve(Quu,Qux)

            Vx = Qx + K[i].T.dot(Quu).dot(k[i]) + K[i].T.dot(Qu) + Qux.T.dot(k[i])
            Vxx = Qxx + K[i].T.dot(Quu).dot(K[i]) + K[i].T.dot(Qux) + Qux.T.dot(K[i])
            Vxx = 0.5*(Vxx+Vxx.T)
        return k, K

    def costs(self,xin,u,terminal=False):
        if terminal == True:
            x=xin-self.xdes
            l = x.T@self.Q_term@x
            lx = self.Q_term@x
            lxx = self.Q_term
            return l, lx, lxx
        else:
            if len(np.shape(u))<2:
                u=u*np.eye(1)
            x=xin-self.xdes
            l = x.T@self.Q@x + u@self.R@u
            lx = self.Q@x
            lxx = self.Q
            lu = self.R@u
            luu = self.R
            lux = np.zeros((self.nact,self.state_size))
            if l>1e6:
                print('stop')
            return l, lx, lxx, lu, luu, lux

    def new_ctrl(self,dyn_model,xs,u,k,K,alpha=1.):
        xs_new = np.zeros_like(xs)
        u_new = np.zeros_like(u)

        l = np.empty(self.tspan)

        xs_new[0] = xs[0].copy()
        for i in range(self.tspan-1):
            u_new[i] = u[i] + alpha*k[i] + K[i].dot(xs_new[i]-xs[i])
            for j in range(len(u_new[i])):
                if dyn_model.active_joints[j]['ctrllim']==1:
                    if u_new[i,j]<dyn_model.active_joints[j]['ctrlrange'][0]:
                        u_new[i,j]=dyn_model.active_joints[j]['ctrlrange'][0]
                    elif u_new[i,j]>dyn_model.active_joints[j]['ctrlrange'][1]:
                        u_new[i,j]=dyn_model.active_joints[j]['ctrlrange'][1]

            self.dyn_model.step_forward(u_new[i])
            xs_new[i+1]=dyn_model.get_state(dyn_model.data)
            l[i], _, _, _, _, _ = self.costs(xs_new[i],u_new[i],terminal=False)
            
        l[-1], _, _ = self.costs(xs_new[-1],u_new[-1],terminal=True)
 
        return xs_new, u_new, l