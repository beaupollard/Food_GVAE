Food Processing Variational Autoencoder (VAE)

This code serves to test the capability of a VAE to linearize the dynamics of a human demonstrator onto a continuous latent space. Once the encoder and decoder are trained on human data, a robot attempts to perform the same task of the human demonstrator. The robot dynamics are encoded onto the latent space using the encoder trained on human data and a new latent space linearized dynamics model is learned for the robot. The intended result is that the robot will be able to efficiently learn the task demonstrated by the humans.

Dependencies:
pytorch
numpy
scipy
matplotlib

All dependencies can be installed and downloaded using pip. I've used Copelliasim to run simulations of a robot mimicking the human task. I will update the repo shortly with scripts to interact with Copelliasim using a python api. 

To get started with the code you have to train the encoder/decoder on human data. The human data is found in data/data_exp_pos.pt. To train on the human data call main_human.py.

Once the encoder/decoder are trained the network learning the robots linearized latent space dynamics model is trained by running main_robot.py 

Running the code can be done all within the main.py script. 