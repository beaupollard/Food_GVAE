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
        self.lr=lr
        self.count=0
        self.kl_weight=10.0
        self.lin_weight=10.0
        self.flatten = nn.Flatten()
        self.latent_dim=latent_dim
        self.enc_out_dim=enc_out_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_height, hidden_layers),
            # nn.Tanh()
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
        )
        self.linear_mu = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )
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

        self.decodersim1= nn.Sequential(
            nn.Linear(latent_dim, latent_dim**2+2*latent_dim),
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

    def reparametrize(self,mu,logstd):
        if self.training:
            return mu+torch.randn_like(logstd)*torch.exp(logstd)
        else:
            return mu

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        mu = self.linear_mu(logits)
        logstd = torch.exp(self.linear_logstd(logits)/2)
        # z = self.reparametrize(mu,logstd)
        return logits, mu, logstd

    def decoder(self,z,inp):
        xhat= self.decoder0(z)
        lin=self.decoder1(z)
        A=torch.reshape(lin[:,:self.latent_dim**2],(len(z),self.latent_dim,self.latent_dim))
            # B=torch.reshape(lin[self.latent_dim**2:2*self.latent_dim**2],(self.latent_dim,1))        
            # K=torch.reshape(lin[2*self.latent_dim**2:],(1,self.latent_dim))            
        B=torch.reshape(lin[:,self.latent_dim**2:self.latent_dim**2+self.latent_dim],(len(z),self.latent_dim,1))   
            

        return xhat, A, B

    def decoder_sim(self,z,inp):
        xhat= self.decoder0(z)
        lin=self.decodersim1(z)
        if len(z.size())==2:
            A=torch.reshape(lin[:,:self.latent_dim**2],(len(z),self.latent_dim,self.latent_dim))
            # B=torch.reshape(lin[self.latent_dim**2:2*self.latent_dim**2],(self.latent_dim,1))        
            # K=torch.reshape(lin[2*self.latent_dim**2:],(1,self.latent_dim))            
            B=torch.reshape(lin[:,self.latent_dim**2:self.latent_dim**2+self.latent_dim],(len(z),self.latent_dim,1))        
            K=torch.reshape(lin[:,self.latent_dim**2+self.latent_dim:],(len(z),1,self.latent_dim))
        else:
            A=torch.reshape(lin[:self.latent_dim**2],(self.latent_dim,self.latent_dim))
            # B=torch.reshape(lin[self.latent_dim**2:2*self.latent_dim**2],(self.latent_dim,1))        
            # K=torch.reshape(lin[2*self.latent_dim**2:],(1,self.latent_dim))            
            B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim],(self.latent_dim,1))        
            K=torch.reshape(lin[self.latent_dim**2+self.latent_dim:],(1,self.latent_dim))               
        
        return xhat, A, B, K

    def decoder_sim_LTI(self,z,inp):
        xhat= self.decoder0(z)
        lin=self.decodersim1(z)
        A=torch.reshape(lin[:,:self.latent_dim**2],(len(z),self.latent_dim,self.latent_dim))
        # B=torch.reshape(lin[self.latent_dim**2:2*self.latent_dim**2],(self.latent_dim,1))        
        # K=torch.reshape(lin[2*self.latent_dim**2:],(1,self.latent_dim))            
        B=torch.reshape(lin[:,self.latent_dim**2:self.latent_dim**2+self.latent_dim],(len(z),self.latent_dim,1))        
        K=torch.reshape(lin[:,self.latent_dim**2+self.latent_dim:],(len(z),1,self.latent_dim))    
        return xhat, A, B, K

    def decoder_sim_ctrl(self,inp):

        lin=self.decodersim2(inp)

        # B=torch.reshape(lin[self.latent_dim**2:2*self.latent_dim**2],(self.latent_dim,1))        
        # K=torch.reshape(lin[2*self.latent_dim**2:],(1,self.latent_dim))            
        B=torch.reshape(lin[:self.latent_dim],(self.latent_dim,1))        
        K=torch.reshape(lin[self.latent_dim:],(1,self.latent_dim))    
        return B, K

    def configure_optimizers(self,lr=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))

    def kl_divergence(self, z, mu, std, mu_prior=[], std_prior=[]):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
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

    def training_step(self, batch,device):
        running_loss=[0.,0.,0.]
        lin_ap=[]
        
        inp=torch.tensor([0],dtype=torch.float).to(device)
        for i in iter(batch):
            self.optimizer.zero_grad()
            x = i[0][:,:-1].to(device)
            y = i[1][:,:-1].to(device)         

            # encode x to get the mu and variance parameters
            x_encoded, mu, std = self.forward(x)

            q=torch.distributions.Normal(mu,std)
            z=q.rsample()

            # decoded
            x_hat, A, B = self.decoder(z,inp)

            y_encoded, muy, stdy = self.forward(y)
            qy=torch.distributions.Normal(muy,stdy)
            ztp1=qy.rsample()  

            ## Calculate the z_(t+1) estimate from linearized model ##
            zout=torch.empty_like(z,requires_grad=False)
            for j in range(zout.size()[0]):
                zout[j,:]=A[j,:,:]@z[j,:]#+B*x[j,-1]
            
            ## Calculate the loss ##
            lin_loss=F.mse_loss(zout,ztp1)*self.lin_weight

            recon_loss = -self.gaussian_likelihood(x_hat, self.log_scale, x[:,:])*100.#F.mse_loss(x_hat,x)#-self.gaussian_likelihood(x_hat, self.log_scale, x)
            kl = self.kl_divergence(z, mu, std)*self.kl_weight
            
            elbo=(kl+recon_loss).mean()+lin_loss

            elbo.backward()

            self.optimizer.step()
            running_loss[0] += np.exp(recon_loss.mean().item()/len(zout))
            running_loss[1] += kl.mean().item()/len(zout)
            running_loss[2] += lin_loss.item()/len(zout)
            lin_ap.append(lin_loss.item())
        self.count+=1
        # self.scheduler.step()
        return running_loss

    def test(self, batch, device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i[0][:,:-1].to(device)
                y = i[1][:,:-1].to(device)  
                # encode x to get the mu and variance parameters
                x_encoded, z, std = self.forward(x[:,:self.enc_out_dim])
                x_hat, A, B = self.decoder(z,inp)
                
                y_encoded, zt1, stdy = self.forward(y[:,:self.enc_out_dim])

                zout=torch.empty_like(z,requires_grad=False)
                for j in range(zout.size()[0]):
                    zout[j,:]=A[j,:,:]@z[j,:]#+B*x[j,-1]
                # decoded
                
        return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy(), zout.detach().numpy(), zt1.detach().numpy()

    def test_sim(self, batch,device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i[0].to(device)
                y = i[1].to(device)   
                # encode x to get the mu and variance parameters
                x_encoded, z, std = self.forward(x[:,:self.enc_out_dim])
                x_hat, A, B, K =self.decoder_sim_LTI(z,inp)
                
                y_encoded, zt1, stdy = self.forward(y[:,:self.enc_out_dim])

                zout=torch.empty_like(z,requires_grad=False)
                u=[]
                for j in range(zout.size()[0]):
                    zout[j,:]=A[j,:,:]@z[j,:]+(B[j,:]*x[j,-1]).flatten()#(A-B@K)@z[j,:]#+B*x[j,-1]
                    u.append(-(K[j,:]@z[j,:]).detach().numpy().item())
                # decoded
                
        return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy(), zout.detach().numpy(), zt1.detach().numpy(), np.array(u)

    def training_sim(self, batch,device):
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
                x_encoded, mu, std = self.forward(x[:,:-1])

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()
                y_encoded, muy, stdy = self.forward(y[:,:-1])
                qy=torch.distributions.Normal(muy,stdy)
                ztp1=qy.rsample()  

            # decoded
            x_hat, A, B, K = self.decoder_sim_LTI(z,inp)

            ## Calculate the z_(t+1) estimate from linearized model ##
            zout=torch.empty_like(z,requires_grad=False)
            for j in range(zout.size()[0]):
                zout[j,:]=A[j,:,:]@z[j,:]+(B[j,:]*x[j,-1]).flatten()
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

    def get_ctrl_from_human(self, human, robot, device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            _, A, B, K =self.decoder_sim(torch.zeros(self.latent_dim,dtype=torch.float).to(device),inp)
            running_loss=[0.,0.,0.]
        
            self.optimizer.zero_grad()
            x = robot[0].to(device) 
            zt1 = torch.tensor(human,dtype=torch.float)

            # encode x to get the mu and variance parameters
            _, zt0, _ = self.forward(x)
            # _, zt1, _ = self.forward(y)
            const=B.T@B
            u=(B.T@(zt1-A@zt0))/const[0]

            # decoded
            # x_hat, _, _, _ = self.decoder_sim(z,inp)
        return u.item(), zt0.detach().numpy()

    def get_ctrl(self, batch, device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            _, A, B, K =self.decoder_sim(torch.zeros(self.latent_dim,dtype=torch.float).to(device),inp)
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i.to(device) 

                # encode x to get the mu and variance parameters
                x_encoded, z, std = self.forward(x)
                zout=torch.empty_like(z,requires_grad=False)
                zout=(A-B@K)@z

                # decoded
                x_hat, _, _, _ = self.decoder_sim(z,inp)
        return x_hat.detach().numpy(), z.detach().numpy(), (-K@z.T).detach().numpy().item()

    def get_ctrl_LQR(self, z_tracked, batch, device=[], prev_err=[]):
        with torch.no_grad():

            inp=torch.tensor([0],dtype=torch.float).to(device)

            R = 1.0*np.ones(1)#np.eye(3)
            Q = 0.1*np.eye(self.latent_dim)
            

            running_loss=[0.,0.,0.]
            
            x = batch[0]
            x_encoded, z, std = self.forward(torch.tensor(x[:-1]))

            zrec=[z.detach().numpy()]
            urec=[]
            for i in range(999):
                x_hat, A, B, _ = self.decoder_sim_LTI(z.reshape((1,self.latent_dim)).type(torch.float),inp)
                K, _, _ = control.dlqr(A[0,:,:],B[0,:,:],Q,R)
                u=-K@(z_tracked[i]-z.detach().numpy())
                if abs(u)>30:
                    u=30*u/abs(u)
                zrec.append((A[0,:,:]@z.type(torch.float)+B[0,:,:]@u).detach().numpy())
                z=A[0,:,:]@z.type(torch.float)+B[0,:,:]@u
                # zrec.append(z.detach().numpy())
                urec.append(u)


            # for i in range(len(x)):
            

            # K, _, _ = control.dlqr(A,B,Q,R)
            # # encode x to get the mu and variance parameters
            # x_encoded, z, std = self.forward(x[:-1])
            # # u = (B.T@(z_human-A@z)/const[0]).detach().numpy()
            # u = -K@((z_tracked-z).T.detach().numpy())

        return zrec

    def project_forward(self, batch, device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            _, A, B, K =self.decoder_sim(torch.zeros(self.latent_dim,dtype=torch.float).to(device),inp)
            running_loss=[0.,0.,0.]
            z_state=[]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i.to(device) 
                
                # encode x to get the mu and variance parameters
                x_encoded, z, std = self.forward(x)
                for j in range(400):
                    zout=(A-B@K)@z
                    z=copy.copy(zout)
                    z_state.append(z.detach().numpy())

                # decoded
                x_hat, _, _, _ = self.decoder_sim(z,inp)
        return z_state

    def project_human_forward(self, batch, device, tspan=800):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            running_loss=[0.,0.,0.]
            zrec=[]
            xrec=[]

            x = batch[0].to(device)
                
            # encode x to get the mu and variance parameters
            x_encoded, z, std = self.forward(x)
            x_hat, A, B = self.decoder(z,inp)
            zrec.append((z).detach().numpy())
            xrec.append(x_hat.detach().numpy())
            for i in range(tspan):
                zrec.append((A@torch.tensor(zrec[-1])).detach().numpy())
                x_hat, _, _ = self.decoder(torch.tensor(zrec[-1]),inp)
                xrec.append(x_hat.detach().numpy())

        return np.array(xrec), np.array(zrec)

    def plot_latent_smooth(self,xinp,yinp,fc=1.):
        fs=1/0.01
          
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        output = signal.filtfilt(b, a, xinp)
        output2 = signal.filtfilt(b, a, yinp)

        return np.array([output,output2])
    
    def forward_difference(self,batch,device):
        for i in iter(batch):
            self.optimizer.zero_grad()
            x = i.to(device) 
            x_encoded, z, std = self.forward(x[:,:-1])
            _, A, B, _ =self.decoder_sim_LTI(z,0)
        return A, B