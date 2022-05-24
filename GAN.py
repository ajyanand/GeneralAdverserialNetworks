import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset
from scipy import linalg
from scipy.stats import entropy
import random
import tqdm
# Resize image to this size
image_size=64
random.seed(40)

# Setting up transforms to resize and normalize 
transform=transforms.Compose([ transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batchsize of dataset
batch_size = 128

# Load STL-10 Dataset
gan_train_dataset = datasets.STL10(root='./stl10_data/', split='train', transform=transform, download=True)
gan_train_loader = torch.utils.data.DataLoader(dataset=gan_train_dataset, batch_size=batch_size, shuffle=True)


class DCGAN_Generator(torch.nn.Module):
    def __init__(self):
        super(DCGAN_Generator,self).__init__()
        self.l1 = torch.nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.l2 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.l3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.l4 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.l5 = torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.layer = torch.nn.Sequential(self.l1, self.bn1, self.relu, self.l2, self.bn2, self.relu, self.l3, self.bn3, self.relu, self.l4, self.bn4, self.relu, self.l5, self.tanh)

    def forward(self, input):
        return self.layer(input)
        #return torch.tanh(self.l4(output))

class DCGAN_Discriminator(torch.nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        self.l1 = torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        self.l2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)       
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.l3 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)   
        self.bn3 = torch.nn.BatchNorm2d(512)
        self.l4 = torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(1024)
        self.l5 = torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer = nn.Sequential(self.l1, self.relu, self.l2, self.bn2, self.relu, self.l3, self.bn3, self.relu , self.l4, self.bn4, self.relu, self.l5, self.sigmoid)

    def forward(self, input):
      return self.layer(input)

torch.autograd.set_detect_anomaly(True)
fake = torch.load('test_case_GAN/fake.pt')
netD = torch.load('test_case_GAN/netD.pt')
real = torch.load('test_case_GAN/real.pt')
netG = torch.load('test_case_GAN/netG.pt')
noise = torch.load('test_case_GAN/noise.pt')
Valid_label = torch.load('test_case_GAN/Valid_label.pt')
Fake_label = torch.load('test_case_GAN/Fake_label.pt')
criterion = torch.load('test_case_GAN/criterion.pt')


def loss_discriminator(D, real, G, noise, Valid_label, Fake_label, criterion):
    '''
    1. Forward real images into the discriminator
    2. Compute loss between Valid_label and dicriminator output on real images
    3. Forward noise into the generator to get fake images
    4. Forward fake images to the discriminator
    5. Compute loss between Fake_label and discriminator output on fake images
    6. sum real loss and fake loss as the loss_D
    7. we also need to output fake images generate by G(noise) for loss_generator computation
    '''

    real = torch.squeeze(real, 1)
    real = torch.squeeze(real, 1)
    real = torch.squeeze(real, 1)
    #print(netG)


    #print("real image SHAPE", real.shape)
    d = D(real)
    #print("Real disc output", d)
    real_output = torch.squeeze(d, 1)
    real_output = torch.squeeze(real_output, 1)
    real_output = torch.squeeze(real_output, 1)

    #print(Valid_label.shape)
    #print(real_output.shape)

    real_loss = criterion(Valid_label, real_output)

    print("nan in noise?", torch.isnan(noise).any())
    fake_gen = G(noise)
    print("nan in output of G?", torch.isnan(fake_gen).any())
    #print(D)
    #print("Generator output SHAPE", fake_gen.shape)
    d2 = D(fake_gen)
    #print("Fake disc output", d2)
    fake_output = torch.squeeze(d2, 1)
    fake_output = torch.squeeze(fake_output, 1)
    fake_output = torch.squeeze(fake_output, 1)


    fake_loss = criterion(Fake_label, fake_output)

    #print(real_loss, fake_loss)
    loss_D = real_loss + fake_loss

    return loss_D, fake_gen

def loss_generator(netD, fake, Valid_label, criterion):
    '''
    1. Forward fake images to the discriminator
    2. Compute loss between valid labels and discriminator output on fake images
    '''

    #print(fake)
    pred = netD(fake)  
    #print("OUT", pred)

    pred = torch.squeeze(pred, 1)
    pred = torch.squeeze(pred, 1)
    pred = torch.squeeze(pred, 1)  
    loss_G = criterion(Valid_label, pred)
    return loss_G


#print(netD)

loss_D, fake_G = loss_discriminator(netD, real, netG, noise, Valid_label, Fake_label, criterion)
torch.save(loss_D, 'test_case_GAN/loss_D.pt')
loss_G = loss_generator(netD, fake, Valid_label, criterion)
torch.save(loss_G, 'test_case_GAN/loss_G.pt')

test_loss_D = torch.load('test_case_GAN/loss_D.pt')
test_loss_G = torch.load('test_case_GAN/loss_G.pt')

print('test case loss_D:', test_loss_D.item())
print('computed loss_D:', loss_D.item())

print('test case loss_G:', test_loss_G.item())
print('computed loss_G:', loss_G.item())



import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of channels
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 128
# Size of feature maps in discriminator
ndf = 128


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Create the generator and discriminator
netG = DCGAN_Generator().to(device)
netD = DCGAN_Discriminator().to(device)

# Apply weight initialization
netG.apply(weights_init)
netD.apply(weights_init)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create latent vector to test the generator performance
fixed_noise = torch.randn(36, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

learning_rate = 0.0002
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

img_list = []
real_img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 100

  
def load_param(num_eps):
  model_saved = torch.load('/content/gan_{}.pt'.format(num_eps))
  netG.load_state_dict(model_saved['netG'])
  netD.load_state_dict(model_saved['netD'])

# GAN Training Loop
for epoch in range(num_epochs):
    for i, data in enumerate(gan_train_loader, 0):
        real = data[0].to(device)
        b_size = real.size(0)
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        Valid_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        Fake_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        netD.zero_grad()
        torch.nn.utils.clip_grad_norm_(netD.parameters(), 1)

        # Function to compute discriminator loss
        loss_D, fake = loss_discriminator(netD, real, netG, noise, Valid_label, Fake_label, criterion)


        # torch.save(fake,'test_case_GAN/fake.pt')
        # torch.save(netD,'test_case_GAN/netD.pt')
        # torch.save(real,'test_case_GAN/real.pt')
        # torch.save(netG,'test_case_GAN/netG.pt')
        # torch.save(noise,'test_case_GAN/noise.pt')
        # torch.save(Valid_label,'test_case_GAN/Valid_label.pt')
        # torch.save(Fake_label,'test_case_GAN/Fake_label.pt')
        # torch.save(criterion,'test_case_GAN/criterion.pt')

        # pdb.set_trace()
        loss_D.backward(retain_graph=True)
        # Update D

        for name, param in netD.named_parameters():
            print(name, param.grad)
            print(name, torch.isfinite(param.grad).all())
        
        optimizerD.step() 

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # Function to compute generator loss

        loss_G = loss_generator(netD, fake, Valid_label, criterion)
        # Calculate gradients for G
        loss_G.backward()
        # Update G

        torch.nn.utils.clip_grad_norm_(netG.parameters(), 1)
        optimizerG.step() 

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, num_epochs, i, len(gan_train_loader),
                     loss_D.item(), loss_G.item()))

        # Save Losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(gan_train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

        

plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

checkpoint = {'netG': netG.state_dict(),
              'netD': netD.state_dict()}
torch.save(checkpoint, 'content/gan_{}.pt'.format(num_epochs))
