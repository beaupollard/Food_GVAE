import numpy as np
from VAE import VAE
import torch

class latent_dynamics():
    def __init__(self,state_dim,action_dim,current_state=[],model_file='./models/swing_up3',traj='./data_swing_val.pt'):
        self.nstate=state_dim
        self.nact=action_dim
        self.device=torch.device("cpu")
        d1=torch.load(traj)
        test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
        self.model=VAE(enc_out_dim=len(d1[0][0])-1,input_height=len(d1[0][0])-1)
        self.model.load_state_dict(torch.load(model_file)) 
        _, self.z_traj, _, _, _, _ = self.model.test_human(test,self.device)
        self.set_current_state(current_state)
        
        self.active_joints=[{'ctrllim':0}]

        
    def get_state(self,data):
        return self.data.detach().numpy()

    def forward_difference(self,eps=[]):
        # mu, _ = self.model.forward(self.current_state)
        _, A, B, _ = self.model.decoder_LTV(self.data.unsqueeze(0))
        return A[0].detach().numpy(), B[0].detach().numpy()
    
    def step_forward(self,u):
        
        _, A, B, _ = self.model.decoder_LTV(self.data.unsqueeze(0))  
        self.data = A[0]@self.data+B[0].flatten()*torch.tensor(u,dtype=torch.float)

    def reset_data(self):
        self.data, _ = self.model.forward(self.current_state)

    def set_current_state(self,current_state):
        self.current_state=current_state
        self.data, _ = self.model.forward(self.current_state)