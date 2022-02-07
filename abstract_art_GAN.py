# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:20:46 2022

@author: Amar Singh
DCGAN Architecture / Implementation inspired by:
Aakash Rao N S
https://jovian.ai/aakashns/06b-anime-dcgan
"""

from IPython.display import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
import random
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

data_dir = "data/Abstract_gallery"
random.seed(53)

# Image Parameters
image_px = 128
batch_size = 128
# First series for stats var is mean for each channel that will be subtracted during norm, 
# second series is std that will be divided during norm
stats = (.5, .5, .5), (.5, .5, .5) 


train_ds = ImageFolder(data_dir, transform=T.Compose([
    T.Resize(image_px),
    T.CenterCrop(image_px),
    T.ToTensor(),
    T.Normalize(*stats)
    ]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

def denormalize(tensors):
    # Following code assumes same value for mean norm and std norm across all channels
    return tensors * stats[1][0] + stats[0][0] 

def show_images(images, nmax=64):
    fig, axis = plt.subplots(figsize=(10,10))
    axis.set_xticks([]); axis.set_yticks([]);
    axis.imshow(make_grid(denormalize(images.detach()[:nmax])).permute(1, 2, 0))
    
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break
    
show_batch(train_dl)

# GPU compatibility
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
# Define discriminator network

discriminator = nn.Sequential(
    # input dimension: 3 x 128 x 128
    nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # output dimension: 128 x 64 x 64

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # output dimension: 256 x 32 x 32
    
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # output dimension: 512 x 16 x 16

    nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.2, inplace=True),
    # output dimension: 1024 x 8 x 8
    
    nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2048),
    nn.LeakyReLU(0.2, inplace=True),
    # output dimension: 2048 x 4 x 4
    
    nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # output dimension: 1 x 1 x 1
    
    nn.Flatten(),
    nn.Sigmoid()
    )


# Define generator network

latent_size = 128
generator = nn.Sequential(
    # input: latent_size x 1 x 1
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32
    
    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    # out: 32 x 64 x 64
    
    nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 128 x 128
    )

#------------------------ Training ------------------------# 

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)

def generate_images(latent_vector = torch.randn(batch_size, latent_size, 1, 1, device=device)):
    latent_vector = torch.randn(batch_size, latent_size, 1, 1, device=device)
    generated_data = generator(latent_vector)
    return generated_data
    
def train(real_data, opt_d, opt_g): 
    #-- Train discriminator network --#
    
    # Clear discriminator gradients
    opt_d.zero_grad()
    
    # Pass real images through discriminator and calculate loss on real images
    real_predictions = discriminator(real_data)
    real_targets = torch.ones(real_data.size(0), 1, device=device)
    real_loss_d = F.binary_cross_entropy(real_predictions, real_targets)
    real_score = torch.mean(real_predictions).item()

    # Use generator to generate images
    generated_data = generate_images()
    
    # Pass generated images through discriminator and calculate loss on generated images
    fake_targets = torch.zeros(generated_data.size(0), 1, device=device)
    fake_predictions = discriminator(generated_data)
    fake_loss_d = F.binary_cross_entropy(fake_predictions, fake_targets)
    fake_score = torch.mean(fake_predictions).item()
    
    # Update discriminator weights and calculate total discriminator loss
    discriminator_loss = real_loss_d + fake_loss_d
    discriminator_loss.backward()
    opt_d.step()
    
    
    #-- Train generator network --#
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate another set of images
    generated_data = generate_images()
    
    # Pass generated images through discriminator (try to trick discriminator)
    predictions = discriminator(generated_data)
    targets = torch.ones(batch_size, 1, device=device)
    generator_loss = F.binary_cross_entropy(predictions, targets)
    
    # Update generator weights
    generator_loss.backward()
    opt_g.step()
    
    #-- Return values --#
    return discriminator_loss.item(), generator_loss.item(), real_score, fake_score


g_dir = "generated_image_set"

os.makedirs(g_dir, exist_ok=True)

# Define function that will generate set of images using same latent vector to track progress as generator weights are adjusted
def save_image_set(index, latent_tensors, show=True):
    generated_images = generator(latent_tensors)
    fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denormalize(generated_images), os.path.join(g_dir, fname), nrow=8)
    print('Saving... ', fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(generated_images.cpu().detach(), nrow=8).permute(1, 2, 0))
    
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_image_set(0, fixed_latent)


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, _ in train_dl:
            loss_d, loss_g, real_score, fake_score = train(real_images, opt_d, opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_image_set(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

lr = 0.0004
epochs = 2

history = fit(epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

# Save the model checkpoints 
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')

Image('./generated/generated-images-0001.png')


