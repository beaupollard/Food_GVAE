Food Processing Variational Autoencoder (VAE)

This code serves to test the capability of a VAE to linearize the dynamics of a system on a low dimensional latent space

The script springmassdamper.py first generates simulation data of multiple spring mass damper systems. This data is then saved into a tensor for training of the VAE. The VAE structure, training, and testing loops are contained within the script VAE.py

Running the code can be done all within the main.py script. 