# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:47:31 2021

@author: KellyUM
"""

from PIL import Image
import numpy as np
import os
import math
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms#, models
from itertools import chain
from copy import deepcopy
import matplotlib as mpl
backend_ =  mpl.get_backend() 
mpl.use("Agg")  # Prevent showing stuff
import matplotlib.pyplot as plt
from mobilefacenet import MobileFaceNet

torch.cuda.empty_cache()
plt.ioff()

###############################################################################

# Change the following paths to use other data.
# Images are expected to be sorted into folders by identity.
dataroot = "data/FRGC_similar/FRGC_croppedportrait"
dataroot_val = "data/FRGC_similar/FRGC_croppedportrait_val"

# Path to the (pretrained) weights of MobileFaceNet.
path_model = 'data/MobileFaceNet_parameters.pt'

# Save the results in the following folder.
newpath = 'results'
if not os.path.exists(newpath):
    os.makedirs(newpath)

###############################################################################

lr = 0.0001                
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
workers = 0
batch_size=64
num_epochs=200
nz = 128 # dimension of the FR latent space
ngf = 32 # number of feature maps
nc = 3   # number of image channels
image_size = 224
crop_size = 34

# beta1, beta2 hyperparameters for Adam optimizer
beta1 = 0.9
beta2 = 0.999

weight_pixel_morph = 1.0
weight_latent_morph = 0.1
epsilon = 0.0 # amount of gaussian noise to add to images

###############################################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def transform():
    return transforms.Compose([
                            transforms.Resize(224),
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])


# calculate a bias that when passed through the 
# sigmoid activation layer outputs an average face.
image_paths = [dataroot + '/' + folder + '/' + im for folder in os.listdir(dataroot) for im in os.listdir(dataroot + '/' + folder)]
initial_bias = transform()( Image.open(image_paths[0]) )
initial_bias_unnorm = initial_bias.new(*initial_bias.size())
initial_bias_unnorm[0, :, :] = initial_bias[0, :, :] * std[0] + mean[0]
initial_bias_unnorm[1, :, :] = initial_bias[1, :, :] * std[1] + mean[1]
initial_bias_unnorm[2, :, :] = initial_bias[2, :, :] * std[2] + mean[2]  
n = 0
for i in np.arange(1,len(image_paths),50):
    im = Image.open(image_paths[i])
    # im = im.resize((image_size,image_size))
    im = transform()( im )
    im_unnorm = im.new(*im.size())
    im_unnorm[0, :, :] = im[0, :, :] * std[0] + mean[0]
    im_unnorm[1, :, :] = im[1, :, :] * std[1] + mean[1]
    im_unnorm[2, :, :] = im[2, :, :] * std[2] + mean[2]  
    initial_bias_unnorm = initial_bias_unnorm*(n+1) + im_unnorm
    initial_bias_unnorm = 1/(n+2) * initial_bias_unnorm
    n+=1
initial_bias = -torch.log((1-initial_bias_unnorm)/initial_bias_unnorm)

# custom dataloader
class MyDataset(Dataset):
    # if single=False image_paths is a list of triplets
    def __init__(self, dataroot=None, train=True): 
        self.dataroot = dataroot
        self.identities = [folder for folder in os.listdir(self.dataroot) for im in os.listdir(self.dataroot + '/' + folder)]
        self.image_paths = [self.dataroot + '/' + folder + '/' + im for folder in os.listdir(self.dataroot) for im in os.listdir(self.dataroot + '/' + folder)]
        self.train = train
    def __getitem__(self, index):
        path0 = self.image_paths[index]
        id0 = self.identities[index]
        im0 = Image.open(path0)
        im0 = transform()(im0)+epsilon*torch.randn(3,image_size,image_size)
        im0 = im0.float() # im0 is a tensor with pixel values between 0 and 1
        return im0, id0
    
    def __len__(self):
        return len(self.image_paths)

# custom weight initialization
def weights_init(m):
    classname = m.__class__.__name__    
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.01) 
        if m.bias is not None:
            m.bias.data.fill_(0)
            print('initialised bias')
        if classname.find('UntiedBias') != -1:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02) 
        nn.init.constant_(m.bias, 0)

dataset = MyDataset(dataroot, train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    
dataset_val = MyDataset(dataroot_val, train=False)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# load pretrained MobileFaceNet
netE = MobileFaceNet()
netE.load_state_dict(torch.load(path_model))
netE.to(device)
for param in netE.parameters():
        param.requires_grad = False # we don't need gradients for the FR network
netE.eval()

# make the supporting Encoder network
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.Enc = nn.Sequential(
            #### Generator_x
            # parameters: in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1
            # UK: state size = (nc) x 224 x 224
            nn.Conv2d(in_channels=nc, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.LeakyReLU(0.02, inplace=True),
            # UK: state size = (ngf) x 112 x 112
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.LeakyReLU(0.02, inplace=True),
            # UK: state size = (ngf) x 56 x 56
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.LeakyReLU(0.02, inplace=True),
            # UK: state size = (ngf) x 28 x 28
            nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            # UK: state size = (ngf*2) x 14 x 14
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True)
            # UK: state size = (ngf*2) x 7 x 7
            )
            
    def forward(self, input):
        out = self.Enc(input)
        return out
netEnc = Encoder(ngpu).to(device)
netEnc.apply(weights_init)

# make the Decoder network
class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.init_size = 7
        self.output_bias = nn.Parameter(initial_bias.clone(), requires_grad=True) #nn.Parameter(torch.zeros(3,image_size,image_size), requires_grad=True)#
        # state size. (nz x 1 x 1) -> (512 x 1 x 1)
        self.linear1 = nn.Linear(nz, 512)
        # state size. (512 x 1 x 1) -> (nz x 7 x 7)
        self.linear2 = nn.Linear(512, (512-ngf*2) * self.init_size ** 2)
        self.D = nn.Sequential(
            #### Generator_x
            # default parameters: in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1
            # state size. 512 x 7 x 7
            nn.Conv2d(in_channels=512, out_channels=ngf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.LeakyReLU(0.02, inplace=True),
            
            nn.Conv2d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*4),
            nn.LeakyReLU(0.02, inplace=True),        
            # state size. (128 x 7 x 7)
            nn.Upsample(scale_factor=2, mode='bicubic'),#(128 x 14 x 14)
            nn.Conv2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            # state size. (64 x 14 x 14)
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.02, inplace=True),
            # state size. (64 x 14 x 14)
            nn.Upsample(scale_factor=2, mode='bicubic'),#(64 x 28 x 28)
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            # state size. (64 x 28 x 28)
            nn.Upsample(scale_factor=2, mode='bicubic'),#(64 x 56 x 56)
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            
            # state size = (64 x 56 x 56)
            nn.Upsample(scale_factor=2, mode='bicubic'),#(64 x 112 x 112)
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            # UK: state size = (ngf) x 112 x 112
            nn.Upsample(scale_factor=2, mode='bicubic'),#(64 x 112 x 112)
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.LeakyReLU(0.02, inplace=True),
            # state size = (ngf*2) x 224 x 224
            nn.Conv2d(in_channels=64, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
            )
            # state size. (nc) x 224 x 224
            
    def forward(self, input):
        # input~z6
        input_z = input[0]
        input_Enc1 = input[1]
        x5 = self.linear1(input_z)
        x4 = self.linear2(x5)
        out_reshaped = x4.view(x4.shape[0], (512-ngf*2), self.init_size, self.init_size)
        initial_input = torch.cat((out_reshaped, input_Enc1),dim=1)        
        x = torch.sigmoid(self.D(initial_input) + self.output_bias) 
        x_normalized = x.new(*x.size())
        x_normalized[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x_normalized[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x_normalized[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
        return x_normalized

netD = Decoder(ngpu).to(device)
netD.apply(weights_init)
optimizerD = optim.Adam(chain(netD.parameters(), netEnc.parameters()), lr=lr, betas=(beta1,beta2))

######################################################################

print('learning rate = ', str(lr))
# for param in netD.parameters():
#     print(param)

criterion = nn.MSELoss(reduction ='mean')
losses = []
losses_pixel_morph = []
losses_latent_morph = []
Scores_val = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.train()
        netEnc.train()
        img, ID = data # get image and identities
        bs = img.size(0)
        img = img.to(device)
        img2 = deepcopy(img)
        img2 = torch.cat((img2[1:],img2[:1])) # translate batch by 1 to get accomplice
        
        optimizerD.zero_grad()
        
        z = F.normalize(netE(F.interpolate(img, size=112, mode='bicubic')))
        z2 = F.normalize(netE(F.interpolate(img2, size=112, mode='bicubic')))
        z_Enc = netEnc(img)
        
        alpha = 0.5*torch.ones(bs).to(device) #torch.rand(bs).to(device)
        alpha_mobileFN = torch.transpose(alpha.expand(nz,bs),0,1)
        z_morph_wc = F.normalize(alpha_mobileFN*z + (1-alpha_mobileFN)*z2)
        
        z_morph = z_morph_wc
        
        morph = netD([z_morph, z_Enc]) # get approximation of worst-case morph by passing wc embedding through decoder
        z_morph_recon = F.normalize(netE(F.interpolate(morph, size=112, mode='bicubic'))) # get latent embedding of morph
        
        # latent loss: we want z_morph_recon to be close to z_morph_wc
        cosine_latent_morph1 = torch.bmm(z_morph_recon.view(bs, 1, nz), z.view(bs, nz, 1))[:,0,:].clamp(-0.999, 0.999)
        theta_latent_morph1 = torch.acos(cosine_latent_morph1) #* 180 / math.pi
        cosine_latent_morph2 = torch.bmm(z_morph_recon.view(bs, 1, nz), z2.view(bs, nz, 1))[:,0,:].clamp(-0.999, 0.999)
        theta_latent_morph2 = torch.acos(cosine_latent_morph2) #* 180 / math.pi
        error_latent_morph = torch.mean(torch.max(theta_latent_morph1, theta_latent_morph2))
                
        error_pixel_morph = criterion(morph,  img) # pixel loss between img and morph
        
        loss = weight_pixel_morph*error_pixel_morph \
                    + weight_latent_morph*error_latent_morph
        loss.backward()
        optimizerD.step()
        
        with torch.no_grad():
            losses.append(loss.item())
            losses_pixel_morph.append(error_pixel_morph.item())
            losses_latent_morph.append(error_latent_morph.item())
    
        if i%500==0:
            with torch.no_grad():
                print('epoch = '+ str(epoch) + ', iteration = ' + str(i), flush=True)
                print('total loss =  %.4f\tpixel loss morph = %.4f\tlatent loss morph = %.4f'
                      %(loss.item(), losses_pixel_morph[-1], losses_latent_morph[-1]), flush=True)
                if epoch%5==0 or epoch==num_epochs-1:
                    z_Enc = netEnc.eval()(img)
                    morph = netD.eval()([z_morph, z_Enc]) # get approximation of worst-case morph by passing wc embedding through decoder
                    
                    morph_unnorm = morph.new(*morph.size())
                    morph_unnorm[:, 0, :, :] = morph[:, 0, :, :] * std[0] + mean[0]
                    morph_unnorm[:, 1, :, :] = morph[:, 1, :, :] * std[1] + mean[1]
                    morph_unnorm[:, 2, :, :] = morph[:, 2, :, :] * std[2] + mean[2]  
                    
                    original = img.detach().cpu()
                    morph = morph_unnorm.detach().cpu()
                    
                    row1 = vutils.make_grid(original, padding=2, normalize=True, nrow=8)
                    row2 = vutils.make_grid(morph, padding=2, normalize=True, nrow=8)
                    
                    fig = plt.figure(figsize=(30,15))
                    plt.subplot(1,2,1)
                    plt.axis("off")
                    plt.title("Original Images")
                    plt.imshow(np.transpose(row1,(1,2,0)))
                    plt.subplot(1,2,2)
                    plt.axis("off")
                    plt.title("Morph Images (from latent space)")
                    plt.imshow(np.transpose(row2,(1,2,0)))
                    
                    plt.savefig(newpath + "/" + "results%s_iteration%s"%(epoch,i))
                    plt.close(fig)
                    del original, morph, morph_unnorm
        del loss, img, z, z2, z_Enc, z_morph_wc, z_morph
        del theta_latent_morph1, theta_latent_morph2
        del cosine_latent_morph1, cosine_latent_morph2
        del error_pixel_morph, error_latent_morph
        torch.cuda.empty_cache() # clear up gpu memory
    if epoch%5==0 or epoch==num_epochs-1:
        fig=plt.figure(2)
        plt.plot(range(len(losses)),losses, alpha=0.7, label = 'total loss')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Decoder Losses')
        plt.savefig(newpath + '/' + 'Loss_Plot')
        plt.close(fig)   
        
        fig=plt.figure(3)
        plt.plot(range(len(losses_pixel_morph)),losses_pixel_morph, alpha=0.7, label = 'pixel loss (morph, img)')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Decoder Losses: Pixel Morph')
        plt.savefig(newpath + '/' + 'Loss_Plot_pixel_morph')
        plt.close(fig)   
        
        fig=plt.figure(4)
        plt.plot(range(len(losses_latent_morph)),losses_latent_morph, alpha=0.7, label = 'latent loss d(z_morph_recon, z_morph_wc)')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Decoder Losses: Latent Morph')
        plt.savefig(newpath + '/' + 'Loss_Plot_latent_morph')
        plt.close(fig)    
        
    Scores_epoch = []
    for i, data in enumerate(dataloader_val, 0): 
        img, ID = data # get image and identities
        bs = img.size(0)
        img = img.to(device)
        z = F.normalize(netE(F.interpolate(img, size=112, mode='bicubic')))
        z2 = torch.cat((z[1:],z[:1]))
        z_Enc = netEnc(img)
        
        alpha = 0.5*torch.ones(bs).to(device)
        alpha_mobileFN = torch.transpose(alpha.expand(nz,bs),0,1)
        z_morph_wc = F.normalize(alpha_mobileFN*z + (1-alpha_mobileFN)*z2)
                
        cosine = torch.bmm(z.view(bs, 1, nz), z_morph_wc.view(bs, nz, 1))[:,0,:].clamp(-0.999, 0.999)
        theta = torch.acos(cosine) * 180 / math.pi
        for j in range(bs):
            Scores_epoch.append(theta[j].item())
    Score_mean_val = np.mean(Scores_epoch)* 180 / math.pi
    Scores_val.append(Score_mean_val)
    Score_mean_train = np.mean(losses_latent_morph[-i:])* 180 / math.pi
    print('mean score on training set = %.3f\tmean score on validation set = %.3f' %(Score_mean_train,Score_mean_val))
    print('diff validation - training = %.3f' %(Score_mean_val - Score_mean_train))       
    if epoch==num_epochs-1:
        torch.save(netD.state_dict(),  newpath + '/D_parameters_' +str(epoch)+'.pt')   
        torch.save(netEnc.state_dict(),  newpath + '/Enc_parameters_' +str(epoch)+'.pt')  


x = [(i+1) *len(losses_latent_morph)/len(Scores_val) for i in range(len(Scores_val))] # fewer iterations for validation set so this ensures the two plots have the same domain

fig=plt.figure(5)
plt.plot(range(len(losses_latent_morph)),[loss for loss in losses_latent_morph], alpha=0.7, label = 'latent loss: training')
plt.plot(x, Scores_val, alpha=0.7, label = 'latent loss: validation')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Latent training loss vs. validation loss (in degrees)')
plt.savefig(newpath + '/' + 'Loss_Plot_latent_val')
plt.close(fig)   