from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np

class VAE(nn.Module):
    def __init__(self, enc_out_dim=3, latent_dim=2, input_height=3,lr=1e-3,hidden_layers=64,rnn_layers=64):
        super(VAE, self).__init__()
        self.lr=lr
        self.count=0
        self.kl_weight=0.4
        self.flatten = nn.Flatten()
        self.latent_dim=latent_dim
        self.rnn_hidden_layers=rnn_layers
        self.enc_out_dim=enc_out_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_height, hidden_layers),
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

        # self.rnn = nn.RNN(latent_dim,rnn_layers,batch_first=True)
        self.rnn = nn.Linear(latent_dim, rnn_layers)
        self.rnn_hidden = nn.Linear(rnn_layers, rnn_layers)

        self.rnn_out= nn.Sequential(
            nn.Tanh(),
            nn.Linear(rnn_layers,latent_dim)
        )
        self.decoder0 = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers),
            nn.Tanh(),#nn.ReLU(),
            nn.Linear(hidden_layers, enc_out_dim),
        )
        self.decoder1= nn.Sequential(
            nn.Linear(latent_dim, latent_dim**2+latent_dim),
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

    def decoder_rnn(self,z,ht_1):
        xhat= self.decoder0(z[:,:2])
        lin=self.rnn(z)+self.rnn_hidden(ht_1)

      
        # A=torch.reshape(lin[:self.latent_dim**2,0],(self.latent_dim,self.latent_dim))
        # B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim,0],(self.latent_dim,))
        return xhat, self.rnn_out(lin), lin
   
    def decoder(self,z):
        xhat= self.decoder0(z)
        lin=self.decoder1(z)
        A=torch.reshape(lin.mean(dim=0)[:self.latent_dim**2],(self.latent_dim,self.latent_dim))
        B=torch.reshape(lin.mean(dim=0)[self.latent_dim**2:self.latent_dim**2+self.latent_dim],(self.latent_dim,))        
        # A=torch.reshape(lin[:self.latent_dim**2,0],(self.latent_dim,self.latent_dim))
        # B=torch.reshape(lin[self.latent_dim**2:self.latent_dim**2+self.latent_dim,0],(self.latent_dim,))
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

    def training_step(self, batch):
        running_loss=[0.,0.,0.]
        lin_ap=[]
        if self.count==200:
            self.lr=self.lr/2
            self.configure_optimizers(lr=self.lr)
            self.count=0
        # elif self.count==500:
        #     self.kl_weight=1.
        for j in iter(batch):
            ht_1=torch.zeros((len(j[:,0,0]),self.rnn_hidden_layers))
            for i in range(1,len(j[0,0,:])-1):
                self.optimizer.zero_grad()
                x = j[:,:,i]
                y = j[:,:,i+1]

                # encode x to get the mu and variance parameters
                x_encoded, mu, std = self.forward(x)

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()
                # z=torch.cat((q.rsample(),ht_1),1)

                
                # decoded
                x_hat, zout, ht_1 = self.decoder_rnn(z,ht_1)

                y_encoded, muy, stdy = self.forward(y)
                qy=torch.distributions.Normal(muy,stdy)
                ztp1=qy.rsample()  

                # zout=torch.empty_like(z,requires_grad=False)
                # for j in range(zout.size()[0]):
                #     zout[j,:]=A@z[j,:]#+B#*x[j][-1]#torch.reshape(torch.reshape(A[0,:,:]@z[j,:],(self.latent_dim,1)))  
                if i==1:
                    lin_loss=F.mse_loss(zout,ztp1)*1.
                else:
                    lin_loss+=F.mse_loss(zout,ztp1)*1.
                # eigval, eigvec=torch.linalg.eig(A)
                # eigs=torch.column_stack((eigval.real,eigvec.real,eigval.imag,eigvec.imag))
                # eigs_gt=torch.tensor([[1.,-1.],[(2)**0.5/2,-(2)**0.5/2],[(2)**0.5/2,(2)**0.5/2],[0.,0.],[0,0],[0,0]],dtype=torch.float).T
                # lin_loss=lin_loss+F.mse_loss(eigs_gt,eigs)*1.

                recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)#F.mse_loss(z,zhat)-F.mse_loss(x_hat,x)#
                kl = self.kl_divergence(z, mu, std)*self.kl_weight
                
                elbo=(kl-recon_loss).mean()+lin_loss

            elbo.backward()

            self.optimizer.step()
            running_loss[0] += recon_loss.mean().item()
            running_loss[1] += kl.mean().item()#F.mse_loss(zout,z).item()
            running_loss[2] += lin_loss.item()
            lin_ap.append(lin_loss.item())
        self.count+=1
        return running_loss
# import matplotlib.pyplot as plt
# plt.plot(z[:,0].detach().numpy())
# plt.plot(zout[:,0].detach().numpy())
# import matplotlib.pyplot as plt
# plt.plot(x[:,0].detach().numpy())
# plt.plot(x_hat[:,0].detach().numpy())

    def test(self, batch):
        with torch.no_grad():
            xest=[]
            zest=[]
            xgt=[]
            zt1=[]
            xestT=[]
            zestT=[]
            xgtT=[]
            zt1T=[]            
            for j in iter(batch):
                ht_1=torch.zeros((len(j[:,0,0]),self.rnn_hidden_layers))
                for i in range(1,len(j[0,0,:])-1):
                    self.optimizer.zero_grad()
                    x = j[:,:,i]
                    y = j[:,:,i+1]

                    # encode x to get the mu and variance parameters
                    x_encoded, mu, std = self.forward(x)

                    q=torch.distributions.Normal(mu,std)
                    z=q.rsample()
                    # z=torch.cat((q.rsample(),ht_1),1)
                    y_encoded, muy, stdy = self.forward(y)
                    qy=torch.distributions.Normal(muy,stdy)
                    ztp1=qy.rsample()  
                        
                    # decoded
                    x_hat, zout, ht_1 = self.decoder_rnn(z,ht_1)
                    xest.append(x_hat.detach().numpy())
                    xgt.append(x.detach().numpy())
                    zest.append(zout.detach().numpy())
                    zt1.append(ztp1.detach().numpy())
                xestT.append(np.array(xest))
                xgtT.append(np.array(xgt))
                zestT.append(np.array(zest))
                zt1T.append(np.array(zt1))
        return xestT, xgtT, zestT, zt1T