from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np
import copy
from scipy import signal
import matplotlib.pyplot as plt
import control 

class VAE(nn.Module):
    def __init__(self, enc_out_dim=4, latent_dim=6, input_height=4,lr=1e-2,hidden_layers=128):
        super(VAE, self).__init__()
        self.lr=lr                  # learning rate
        self.count=0                # counter
        self.kl_weight=10.0         # KL divergence weight
        self.lin_weight=10.0        # linear transition approximation weight
        self.recon_weight=100.0     # Reconstruction weight
        self.flatten = nn.Flatten() # Flatten array operation
        self.latent_dim=latent_dim  # Dimension of latent space
        self.enc_out_dim=enc_out_dim# Dimension of encoder output array
        
        ## First set of layers in encoder ##
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_height, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
        )
        ## Takes input from first encoder layer and outputs distribution mean ##
        self.linear_mu = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
        )
        ## Takes input from first encoder layer and outputs distribution standard deviation ##
        self.linear_logstd = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            nn.ReLU()
        )

        self.decoder0 = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, enc_out_dim),
        )
        self.decoder1= nn.Sequential(
            nn.Linear(latent_dim, latent_dim**2+latent_dim),
            # nn.Tanh(),#ReLU(),
        )
        self.decoder1_LTI= nn.Sequential(
            nn.Linear(1, latent_dim**2+latent_dim),
            # nn.Tanh(),#ReLU(),
        )
        self.decodersim1= nn.Sequential(
            nn.Linear(latent_dim, latent_dim**2+2*latent_dim),
            # nn.Tanh(),#ReLU(),
        )
        self.decodersim1_LTI= nn.Sequential(
            nn.Linear(1, latent_dim**2+2*latent_dim),
            # nn.Tanh(),#ReLU(),
        )
        self.decodersim2= nn.Sequential(
            nn.Linear(1, 2*latent_dim),
            # nn.Tanh(),#ReLU(),
        )
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.optimizer=self.configure_optimizers(lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        '''Takes in the state x and returns the encoded distribution mean and standard deviation
        Inputs:
        x [Batch size, state space dim]
        Outputs:
        '''
        logits = self.linear_relu_stack(x)
        mu = self.linear_mu(logits)
        logstd = torch.exp(self.linear_logstd(logits)/2)

        return mu, logstd

    def decoder_LTI(self,z,inp):
        '''Takes in a randomly sampled latent space point and outputs the recreation \tilde{x} and 
        approximate state transition matrix \tilde{A} and \tilde{B}'''
        xhat= self.decoder0(z)
        lin=self.decoder1_LTI(inp)
        A=torch.reshape(lin[:self.latent_dim**2],(self.latent_dim,self.latent_dim))
        B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim],(self.latent_dim,)) 
        return xhat, A, B

    def decoder_LTV(self,z):
        '''Takes in a randomly sampled latent space point and outputs the recreation \tilde{x} and 
        approximate state transition matrix \tilde{A} and \tilde{B}'''
        xhat= self.decoder0(z)
        lin=self.decoder1(z)
        A=torch.reshape(lin[:,:self.latent_dim**2],(len(z),self.latent_dim,self.latent_dim))       
        B=torch.reshape(lin[:,self.latent_dim**2:self.latent_dim**2+self.latent_dim],(len(z),self.latent_dim,1))   
        return xhat, A, B

    def decoder_sim_LTV(self,z):
        '''Takes in a sampled latent space point z and returns the linearized dynamic model at that point
        Inputs:
        z [Batch size, latent space dim]
        Outputs:
        A [Batch size, latent space dim, latent space dim]
        B [Batch size, latent space dim, 1]
        K [Batch size, 1, latent space dim]'''        
        xhat= self.decoder0(z)
        lin=self.decodersim1(z)
        if len(z.size())==2:
            A=torch.reshape(lin[:,:self.latent_dim**2],(len(z),self.latent_dim,self.latent_dim))         
            B=torch.reshape(lin[:,self.latent_dim**2:self.latent_dim**2+self.latent_dim],(len(z),self.latent_dim,1))        
            K=torch.reshape(lin[:,self.latent_dim**2+self.latent_dim:],(len(z),1,self.latent_dim))
        else:
            A=torch.reshape(lin[:self.latent_dim**2],(self.latent_dim,self.latent_dim))      
            B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim],(self.latent_dim,1))        
            K=torch.reshape(lin[self.latent_dim**2+self.latent_dim:],(1,self.latent_dim))               
        
        return xhat, A, B, K

    def decoder_sim_LTI(self,z,inp):
        '''Takes in a sampled latent space point z and returns the linearized dynamic model at that point
        Inputs:
        z [Batch size, latent space dim]
        inp [0]
        Outputs:
        A [Batch size, latent space dim, latent space dim]
        B [Batch size, latent space dim, 1]
        K [Batch size, 1, latent space dim]'''
        xhat= self.decoder0(z)
        lin=self.decodersim1_LTI(inp)
        A=torch.reshape(lin[:self.latent_dim**2],(self.latent_dim,self.latent_dim))          
        B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim],(self.latent_dim,1))        
        K=torch.reshape(lin[self.latent_dim**2+self.latent_dim:],(1,self.latent_dim))    
        return xhat, A, B, K

    def configure_optimizers(self,lr=1e-4):
        '''Configures the optimizer
        Input: learning rate'''
        return torch.optim.Adam(self.parameters(), lr=lr)

    def gaussian_likelihood(self, x_hat, logscale, x):
        '''Takes in the decoded latent space distribution mean \hat{x}t and the current state xt. It outputs the 
        likelihood that xt is sampled from the decoded distribution \hat{x}t'''
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))

    def kl_divergence(self, z, mu, std, mu_prior=[], std_prior=[]):
        '''Takes in the latent space distribution parameters (mu, stc) and the sampled latent state (z).
        Outputs the KL divergence of the latent space distribution and the target distribution. If mu_prior
        and std_prior are not specified then the distrubition is compared to a zero mean normal with 
        standard deviation of one.'''
        if len(mu_prior)==0:
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        else:
            p = torch.distributions.Normal(mu_prior, std_prior)
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
        # return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def training_human(self,batch,device,LTI=True):
        running_loss=[0.,0.,0.]
        lin_ap=[]
        
        inp=torch.tensor([0],dtype=torch.float).to(device)
        for i in iter(batch):
            self.optimizer.zero_grad()
            x = i[0].to(device) # x^t
            y = i[1].to(device) # x^t+1        

            # Encode state x to get the latent space mu and variance parameters
            mu, std = self.forward(x)

            # Randomly sample a point from the latent space distribution
            q=torch.distributions.Normal(mu,std)
            z=q.rsample()

            # decoded
            if LTI==True:
                x_hat, A, B = self.decoder_LTI(z,inp)
                ## Calculate the z_(t+1) estimate from linearized model ##
                zout=torch.empty_like(z,requires_grad=False)
                for j in range(zout.size()[0]):
                    zout[j,:]=A@z[j,:]
            else:
                x_hat, A, B = self.decoder_LTV(z)
                ## Calculate the z_(t+1) estimate from linearized model ##
                zout=torch.empty_like(z,requires_grad=False)
                for j in range(zout.size()[0]):
                    zout[j,:]=A[j,:,:]@z[j,:]

            muy, stdy = self.forward(y)
            qy=torch.distributions.Normal(muy,stdy)
            ztp1=qy.rsample()  
            
            ## Calculate the loss ##

            lin_loss=F.mse_loss(zout,ztp1)*self.lin_weight  # State transition matrix loss

            recon_loss = -self.gaussian_likelihood(x_hat, self.log_scale, x[:,:])*self.recon_weight # recreation loss
            kl = self.kl_divergence(z, mu, std)*self.kl_weight  # KL divergence
            
            elbo=(kl+recon_loss).mean()+lin_loss # Sum the losses

            elbo.backward() # Propogate losses backwards

            self.optimizer.step()
            running_loss[0] += np.exp(recon_loss.mean().item()/len(zout))
            running_loss[1] += kl.mean().item()/len(zout)
            running_loss[2] += lin_loss.item()/len(zout)
            lin_ap.append(lin_loss.item())
        self.count+=1
        # self.scheduler.step()
        return running_loss

    def test_human(self, batch, device, LTI=True):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i[0].to(device)
                y = i[1].to(device)  
                # encode x to get the mu and variance parameters
                z, std = self.forward(x[:,:self.enc_out_dim])
                # decoded
                if LTI==True:
                    x_hat, A, B = self.decoder_LTI(z,inp)
                    zout=torch.empty_like(z,requires_grad=False)
                    for j in range(zout.size()[0]):
                        zout[j,:]=A@z[j,:]
                else:
                    x_hat, A, B = self.decoder_LTV(z)
                    zout=torch.empty_like(z,requires_grad=False)
                    for j in range(zout.size()[0]):
                        zout[j,:]=A[j,:,:]@z[j,:]
                zt1, stdy = self.forward(y[:,:self.enc_out_dim])
                
        return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy(), zout.detach().numpy(), zt1.detach().numpy()

    def test_sim(self, batch,device,LTI=True):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i[0].to(device)
                y = i[1].to(device)   
                # encode x to get the mu and variance parameters
                z, std = self.forward(x[:,:self.enc_out_dim])
                zt1, stdy = self.forward(y[:,:self.enc_out_dim])
                if LTI==True:
                    x_hat, A, B, K =self.decoder_sim_LTI(z,inp)
                    zout=torch.empty_like(z,requires_grad=False)
                    u=[]
                    for j in range(zout.size()[0]):
                        zout[j,:]=A@z[j,:]+(B*x[j,-1]).flatten()
                        u.append(-(K@z[j,:]).detach().numpy().item())       
                else:
                    x_hat, A, B, K =self.decoder_sim_LTV(z)
                    zout=torch.empty_like(z,requires_grad=False)
                    u=[]
                    for j in range(zout.size()[0]):
                        zout[j,:]=A[j,:,:]@z[j,:]+(B[j,:]*x[j,-1]).flatten()
                        u.append(-(K[j,:]@z[j,:]).detach().numpy().item())                
                
        return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy(), zout.detach().numpy(), zt1.detach().numpy(), np.array(u)

    def training_sim(self, batch,device,LTI=True):
        running_loss=[0.,0.,0.]
        lin_ap=[]
        
        inp=torch.tensor([0],dtype=torch.float).to(device)
        # _, A_human, _ =self.decoder(torch.zeros(self.latent_dim,dtype=torch.float).to(device),inp)

        for i in iter(batch):
            self.optimizer.zero_grad()
            
            x = i[0].to(device)
            y = i[1].to(device)            
            with torch.no_grad():
                # encode x to get the mu and variance parameters
                mu, std = self.forward(x[:,:-1])

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()
                muy, stdy = self.forward(y[:,:-1])
                qy=torch.distributions.Normal(muy,stdy)
                ztp1=qy.rsample()  

                z, std = self.forward(x[:,:self.enc_out_dim])
                zt1, stdy = self.forward(y[:,:self.enc_out_dim])
            
            if LTI==True:
                x_hat, A, B, K =self.decoder_sim_LTI(z,inp)
                zout=torch.empty_like(z,requires_grad=False)
                u=[]
                for j in range(zout.size()[0]):
                    zout[j,:]=A@z[j,:]+(B*x[j,-1]).flatten()
                    u.append(-(K@z[j,:]).detach().numpy().item())       
            else:
                x_hat, A, B, K =self.decoder_sim_LTV(z)
                zout=torch.empty_like(z,requires_grad=False)
                u=[]
                for j in range(zout.size()[0]):
                    zout[j,:]=A[j,:,:]@z[j,:]+(B[j,:]*x[j,-1]).flatten()
                    u.append(-(K[j,:]@z[j,:]).detach().numpy().item())     

            # eig_loss=F.mse_loss(torch.linalg.eig(A-B@K)[1].real,torch.linalg.eig(A_human)[1].real)*1.0+F.mse_loss(torch.linalg.eig(A-B@K)[1].imag,torch.linalg.eig(A_human)[1].imag)*1.0
            kl_trans_loss=self.kl_divergence(zout, mu, std, mu_prior=muy, std_prior=stdy)
            ## Calculate the loss ##
            lin_loss=F.mse_loss(zout,ztp1)*1.
            # ctrl_loss=F.mse_loss(-(K@z.T).flatten(),x[:,-1])*1.
            elbo=lin_loss+kl_trans_loss.mean()
            # elbo=(kl+recon_loss).mean()+lin_loss

            elbo.backward()

            self.optimizer.step()
            # running_loss[0] += ctrl_loss.item()/len(zout)#np.exp(recon_loss.mean().item()/len(zout))
            running_loss[0] += lin_loss.item()/len(zout)
            running_loss[1] += (kl_trans_loss.mean()).item()/len(zout)
            # running_loss[2] += ctrl_loss.item()/len(zout)
            # running_loss[2] += lin_loss.item()#/len(zout)
            # lin_ap.append(lin_loss.item())
        self.count+=1
        # self.scheduler.step()
        return running_loss

    def get_ctrl(self, batch, device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            _, A, B, K =self.decoder_sim(torch.zeros(self.latent_dim,dtype=torch.float).to(device),inp)
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i.to(device) 

                # encode x to get the mu and variance parameters
                z, std = self.forward(x)
                zout=torch.empty_like(z,requires_grad=False)
                zout=(A-B@K)@z

                # decoded
                x_hat, _, _, _ = self.decoder_sim(z,inp)
        return x_hat.detach().numpy(), z.detach().numpy(), (-K@z.T).detach().numpy().item()

    def get_ctrl_LQR(self, z_tracked, batch, device=[], prev_err=[],LTI=True):
        with torch.no_grad():

            inp=torch.tensor([0],dtype=torch.float).to(device)

            R = 1.0*np.ones(1)#np.eye(3)
            Q = 0.1*np.eye(self.latent_dim)
            

            running_loss=[0.,0.,0.]
            
            x = batch[0]
            z, std = self.forward(torch.tensor(x[:-1]))

            zrec=[z.detach().numpy()]
            urec=[]
            for i in range(999):
                if LTI==True:
                    x_hat, A, B, _ = self.decoder_sim_LTI(z.reshape((1,self.latent_dim)).type(torch.float),inp)
                else:
                    x_hat, A, B, _ = self.decoder_sim_LTV(z.reshape((1,self.latent_dim)).type(torch.float))
                K, _, _ = control.dlqr(A[0,:,:],B[0,:,:],Q,R)
                u=-K@(z_tracked[i]-z.detach().numpy())
                if abs(u)>30:
                    u=30*u/abs(u)
                zrec.append((A[0,:,:]@z.type(torch.float)+B[0,:,:]@u).detach().numpy())
                z=A[0,:,:]@z.type(torch.float)+B[0,:,:]@u
                # zrec.append(z.detach().numpy())
                urec.append(u)

        return zrec

    def plot_latent_smooth(self,xinp,yinp,fc=1.,fs=100.):
        '''Takes in x and y data, the cutoff frequency (fc) and the measurement frequency (fs) and 
        outputs the filtered signals'''
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        output = signal.filtfilt(b, a, xinp)
        output2 = signal.filtfilt(b, a, yinp)

        return np.array([output,output2])
    
    def forward_difference(self,batch,device):
        for i in iter(batch):
            self.optimizer.zero_grad()
            x = i.to(device) 
            z, std = self.forward(x[:,:-1])
            _, A, B, _ =self.decoder_sim_LTI(z,0)
        return A, B