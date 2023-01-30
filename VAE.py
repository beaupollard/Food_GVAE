from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np

class VAE(nn.Module):
    def __init__(self, enc_out_dim=2, latent_dim=2, input_height=2,lr=1e-3,hidden_layers=128):
        super(VAE, self).__init__()
        self.lr=lr
        self.count=0
        self.kl_weight=0.1
        self.flatten = nn.Flatten()
        self.latent_dim=latent_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_height, hidden_layers),
            # nn.Tanh()
            nn.ReLU(),
        )
        self.linear_mu = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )
        self.linear_logstd = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )

        self.decoder0 = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers),
            nn.Tanh(),#nn.ReLU(),
            nn.Linear(hidden_layers, enc_out_dim),
        )
        self.decoder1= nn.Sequential(
            nn.Linear(1, latent_dim**2+latent_dim),
            # nn.Tanh(),#ReLU(),
        )

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.optimizer=self.configure_optimizers(lr=lr)

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
        lin=self.decoder1(inp)
        A=torch.reshape(lin[:self.latent_dim**2],(self.latent_dim,self.latent_dim))
        B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim],(self.latent_dim,))        

        return xhat, A, B


    def configure_optimizers(self,lr=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
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
            x = i[0].to(device)
            y = i[1].to(device)            

            # encode x to get the mu and variance parameters
            x_encoded, mu, std = self.forward(x[:,2:-3])

            q=torch.distributions.Normal(mu,std)
            z=q.rsample()

            # decoded
            x_hat, A, B = self.decoder(z,inp)

            y_encoded, muy, stdy = self.forward(y[:,2:-3])
            qy=torch.distributions.Normal(muy,stdy)
            ztp1=qy.rsample()  

            ## Calculate the z_(t+1) estimate from linearized model ##
            zout=torch.empty_like(z,requires_grad=False)
            for j in range(zout.size()[0]):
                zout[j,:]=A@z[j,:]+B*x[j,-1]
            
            ## Calculate the loss ##
            lin_loss=F.mse_loss(zout,ztp1)*1.0

            recon_loss = -self.gaussian_likelihood(x_hat, self.log_scale, x[:,2:-3])
            kl = self.kl_divergence(z, mu, std)*self.kl_weight
            
            elbo=(kl+recon_loss).mean()+lin_loss

            elbo.backward()

            self.optimizer.step()
            running_loss[0] += recon_loss.mean().item()
            running_loss[1] += kl.mean().item()
            running_loss[2] += lin_loss.item()
            lin_ap.append(lin_loss.item())
        self.count+=1
        return running_loss

    def test(self, batch,device):
        with torch.no_grad():
            inp=torch.tensor([0],dtype=torch.float).to(device)
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x = i[0].to(device)
                y = i[1].to(device)   
                # encode x to get the mu and variance parameters
                x_encoded, mu, std = self.forward(x[:,2:-3])

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()

                # decoded
                x_hat, A, B = self.decoder(z,inp)
        return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy()