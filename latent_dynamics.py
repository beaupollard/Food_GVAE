import numpy as np
from VAE import VAE
import torch

class latent_dynamics():
    def __init__(self,state_dim,action_dim,current_state=[],model=[]):
        self.nstate=state_dim
        self.nact=action_dim
        self.device=torch.device("cpu")
        # d1=torch.load(traj)
        # test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
        # self.model=VAE(enc_out_dim=len(d1[0][0])-1,input_height=len(d1[0][0])-1)
        # self.model.load_state_dict(torch.load(model_file)) 
        self.model=model
        # _, self.z_traj, _, _, _, _ = self.model.test_human(test,self.device)
        self.set_current_state(current_state)
        self.z_traj, self.u_traj=self.model.sim_rob_rollout(self.device,iters=990,z_init=self.data)
        self.z_traj=self.z_traj.detach().numpy()

        self.active_joints=[{'ctrllim':1, 'ctrlrange':[-20.,20.]}]

        
    def get_state(self,data):
        return self.data.detach().numpy()

    def forward_difference(self,eps=[]):
        # mu, _ = self.model.forward(self.current_state)
        _, A, B, _, _, _ = self.model.decoder_sim_LTV(self.data.unsqueeze(0))
        return A[0].detach().numpy(), B[0].detach().numpy()
    
    def step_forward(self,u):
        
        _, A, B, O, K, _ = self.model.decoder_sim_LTV(self.data.unsqueeze(0))  
        self.data = A[0]@self.data+B[0].flatten()*torch.tensor(u,dtype=torch.float) + K[0]

    def reset_data(self):
        self.data, _ = self.model.forward(self.current_state)

    def set_current_state(self,current_state):
        self.current_state=current_state
        self.data, _ = self.model.forward(self.current_state)