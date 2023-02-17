# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:47:31 2021

@author: KellyUM
"""


from PIL import Image
#from keras_vggface.vggface import VGGFace
#from keras.models import Model
import numpy as np
import os
import math
import random
import torch
#import torchvision
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, models
from torch.autograd import grad as torch_grad
from itertools import chain
from copy import deepcopy
import matplotlib as mpl
import dlib
backend_ =  mpl.get_backend() 
mpl.use("Agg")  # Prevent showing stuff
import matplotlib.pyplot as plt
from mobilefacenet import MobileFaceNet

plt.ioff()

dataroot_val =  "data/FRGC_similar/FRGC_croppedportrait_val"
path_eval = 'results'

path_D = path_eval+'/D_parameters_199.pt'
path_Enc = path_eval+'/Enc_parameters_199.pt'


save_reconstructions = True

path_model = 'data/MobileFaceNet_parameters.pt'
                
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
workers = 0
batch_size=8
num_epochs=200

nz = 128 # only using the pretrained part of InceptionResnet
#nz = 256 # when using the last trained layer as well
ngf = 32
nc = 3
image_size = 224
crop_size = 34

epsilon = 0.0 # amount of gaussian noise to add to images


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def transform():
    return transforms.Compose([
                            transforms.Resize(224),
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])

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
        
        if self.train:
            indices_ref = [i for i, x in enumerate(self.identities) if (x == id0 and i!=index)]
            index_ref = indices_ref[random.randint(0, len(indices_ref)-1)]
            path0_ref = self.image_paths[index_ref]
            im0_ref = Image.open(path0_ref)
            im0_ref = transform()(im0_ref)+epsilon*torch.randn(3,image_size,image_size)
            im0_ref = im0_ref.float()
            if not id0==self.identities[index_ref]:
                print(path0, path0_ref)     
            return im0, id0, im0_ref
        else:
            return im0, path0.replace(self.dataroot ,'')
    
    def __len__(self):
        return len(self.image_paths)

dataset_val = MyDataset(dataroot_val, train=False)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

netE = MobileFaceNet()
netE.load_state_dict(torch.load(path_model))
netE.to(device)
for param in netE.parameters():
        param.requires_grad = False
netE.eval()

def get_score_MobFN(z1, z2):
    cosine = torch.bmm(z1.view(1, 1, nz), z2.view(1, nz, 1))[:, 0, :].clamp(-0.999, 0.999)
    theta = torch.acos(cosine) * 180 / math.pi
    return theta


model = models.vgg16(pretrained=False)
model.classifier = nn.Sequential(*[model.classifier[i] for i in range(5)])
model.load_state_dict(torch.load('data/vgg_pretrained.pt'))
netE2 = model.to(device)
for param in netE2.parameters():
        param.requires_grad = False
netE2.eval()

def get_emb(x):
    input_dense = netE2.features(x).permute(2,3,1,0)
    input_dense = input_dense.flatten(start_dim=0, end_dim=2)
    input_dense = input_dense.transpose(1,0)
    z = netE2.classifier(input_dense)#.cpu().numpy()
    z_norm = F.normalize(z)
    return z_norm

def get_score_vgg(z1, z2):
    d = torch.sum((z1-z2)**2, dim=1)**0.5
    return d


## Make Encoder and Decoder networks
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

# make the Decoder network
class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.init_size = 7
        self.output_bias = nn.Parameter(torch.zeros(3,image_size,image_size), requires_grad=True)#
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
netEnc = Encoder(ngpu).to(device)


######################################################################

# for param in netD.parameters():
#     print(param)

netD.load_state_dict(torch.load(path_D))
netEnc.load_state_dict(torch.load(path_Enc))


netD.eval()
netEnc.eval()

Embeddings = [[]]*5
Embeddings_Enc = [[]]*5
Embeddings_vgg = [[]]*5
Names = []
D_gen = []
D_imp = []
with torch.no_grad():
    n=0
    for i, data in enumerate(dataloader_val, 0):
        
        img, ID = data # get image and identities
        bs = img.size(0)
        img = img.to(device)       
        
        z = F.normalize(netE.eval()(F.interpolate(img, size=112, mode='bicubic')))
        z_Enc = netEnc.eval()(img)
        z_vgg = get_emb(img)
        
        Embeddings[n:n+bs] = z.cpu().detach()
        Embeddings_Enc[n:n+bs] = z_Enc.cpu().detach()
        Embeddings_vgg[n:n+bs] = z_vgg.cpu().detach()
        Names[n:n+bs] = [identity.replace('.JPG','.jpg') for identity in ID]
        
        n+=bs
    

    
gan_morphs = open("data/GAN_morph_names.txt",'r').read()
gan_morphs_list = gan_morphs.split("\n")[:-1]
## Do some optimisation steps to improve the worst-case approximations

print('calculated all embeddings, now generate morphs', flush=True)
criterion = nn.MSELoss(reduction ='mean')

for n_opt in [200] :

    scores_wc_est = []
    scores_wc_est_max = []
    scores_wc = []
    vgg_wc_est_max = []
    vgg_wc = []
    
    for n, morph_name in enumerate(gan_morphs_list):
      # if n%50==0:
        print(morph_name)
        id1 = morph_name[:morph_name.find('-vs-')]
        id2 = morph_name[morph_name.find('-vs-')+4:-4]
        
        index1 = Names.index('/'+id1[:5]+'/'+id1+'.jpg')
        index2 = Names.index('/'+id2[:5]+'/'+id2+'.jpg')
        
        z1 = Embeddings[index1].unsqueeze(0)
        z2 = Embeddings[index2].unsqueeze(0)
        
        z1_Enc = Embeddings_Enc[index1].unsqueeze(0)
        z2_Enc = Embeddings_Enc[index2].unsqueeze(0)
    
        z1_vgg = Embeddings_vgg[index1].unsqueeze(0)
        z2_vgg = Embeddings_vgg[index2].unsqueeze(0)    
        
        z_morph_wc = F.normalize((z1+z2)/2)
        z_morph_init1 = F.normalize((z1+z2)/2)
        z_morph_init2 = F.normalize((z1+z2)/2)
        z_morph_init1.requires_grad=True
        z_morph_init2.requires_grad=True
        for i in range(n_opt):
            morph1 = netD.eval()([z_morph_init1.to(device), z1_Enc])
            morph2 = netD.eval()([z_morph_init2.to(device), z2_Enc])
            z_morph1 = F.normalize(netE(F.interpolate(morph1, size=112, mode='bicubic')))
            z_morph2 = F.normalize(netE(F.interpolate(morph2, size=112, mode='bicubic')))
            loss1 = criterion(z_morph1,z_morph_wc)
            loss2 = criterion(z_morph2,z_morph_wc)
            # print(loss)
            gradients_z1 = torch_grad(outputs=loss1, inputs=z_morph_init1,
                                grad_outputs=torch.ones(loss1.size()).to(device),
                                create_graph=True, retain_graph=True)[0]
            gradients_z2 = torch_grad(outputs=loss2, inputs=z_morph_init2,
                                grad_outputs=torch.ones(loss2.size()).to(device),
                                create_graph=True, retain_graph=True)[0]
            z_morph_init1 = z_morph_init1-gradients_z1.detach()      
            z_morph_init2 = z_morph_init2-gradients_z2.detach()      
        
        morph1 = netD.eval()([z_morph_init1, z1_Enc])
        morph2 = netD.eval()([z_morph_init2, z2_Enc])
               
        z_morph1 = F.normalize(netE(F.interpolate(morph1, size=112, mode='bicubic')))
        z_morph2 = F.normalize(netE(F.interpolate(morph2, size=112, mode='bicubic')))
    
        
        # save recontructed image
        if not os.path.exists(path_eval+'/morphs_200/'+id1[:5]+'_'+id2[:5]):
            os.makedirs(path_eval+'/morphs_200/'+id1[:5]+'_'+id2[:5])
        
        if save_reconstructions:
            morph1_unnorm = torch.zeros(morph1.size())
            morph1_unnorm[:, 0, :, :] = morph1[:, 0, :, :] * std[0] + mean[0]
            morph1_unnorm[:, 1, :, :] = morph1[:, 1, :, :] * std[1] + mean[1]
            morph1_unnorm[:, 2, :, :] = morph1[:, 2, :, :] * std[2] + mean[2] 
            morph1_unnorm_np = np.transpose(
                        (255*morph1_unnorm[0].cpu().detach().numpy()).astype('uint8'), (1, 2, 0))
            morph2_unnorm = torch.zeros(morph2.size())
            morph2_unnorm[:, 0, :, :] = morph2[:, 0, :, :] * std[0] + mean[0]
            morph2_unnorm[:, 1, :, :] = morph2[:, 1, :, :] * std[1] + mean[1]
            morph2_unnorm[:, 2, :, :] = morph2[:, 2, :, :] * std[2] + mean[2] 
            morph2_unnorm_np = np.transpose(
                        (255*morph2_unnorm[0].cpu().detach().numpy()).astype('uint8'), (1, 2, 0))
            plt.imsave(path_eval+'/morphs_200/'+id1[:5]+'_'+id2[:5]+'/'+id1+'_'+id2 +'_1.jpg', morph1_unnorm_np)
            plt.imsave(path_eval+'/morphs_200/'+id1[:5]+'_'+id2[:5]+'/'+id1+'_'+id2 +'_2.jpg', morph2_unnorm_np)
        
        
        
        # MobileFaceNet scores
        theta11 = get_score_MobFN(z1, z_morph1)
        theta12 = get_score_MobFN(z_morph1, z2)
        theta21 = get_score_MobFN(z1, z_morph2)
        theta22 = get_score_MobFN(z_morph2, z2)       
        
        theta_wc = get_score_MobFN(z_morph_wc, z1)
        
        scores_wc_est.append(theta11.item())
        scores_wc_est.append(theta12.item())
        scores_wc_est.append(theta21.item())
        scores_wc_est.append(theta22.item())
        
        scores_wc_est_max.append(max(theta11.item(),theta12.item()))
        scores_wc_est_max.append(max(theta21.item(),theta22.item()))
        
        scores_wc.append(theta_wc.item())
        
        ## VGG16 scores
        z_morph1_vgg = get_emb(morph1)
        z_morph2_vgg = get_emb(morph2)
        z_wc_vgg = F.normalize((z1_vgg+z2_vgg)/2)
        
        d11 = get_score_vgg(z1_vgg,z_morph1_vgg)
        d12 = get_score_vgg(z2_vgg,z_morph1_vgg)
        d21 = get_score_vgg(z1_vgg,z_morph2_vgg)
        d22 = get_score_vgg(z2_vgg,z_morph1_vgg)
        d_wc = get_score_vgg(z1_vgg,z_wc_vgg)
        
        vgg_wc_est_max.append(max(d11.item(), d12.item()))
        vgg_wc_est_max.append(max(d21.item(), d22.item()))
        vgg_wc.append(d_wc.item())
              
    print('number of optimisation steps = ', n_opt)
    print('mean of max(score(morph,ref1),(morph,ref2) = ', np.round(np.mean(scores_wc_est_max),2))
    print('VGG: mean of max(score(morph,ref1),(morph,ref2) = ', np.round(np.mean(vgg_wc_est_max),3), flush=True)
    
fig=plt.figure()
plt.hist(scores_wc, bins=50, density=True, color='lightgreen', alpha=0.5, label = 'worst-case morph scores')
plt.hist(scores_wc_est, bins=100, density=True, color='darkgreen', alpha=0.5, label = 'estimated worst-case morph scores')
plt.xlabel('Angle')
plt.xlim(25, 70)
plt.ylabel('frequency')
plt.title('Score(morph, id1)')
plt.legend()
plt.savefig(path_eval+'/Angles_wc_Histogram_200')
plt.close(fig)
    
fig=plt.figure()
plt.hist(scores_wc, bins=50, density=True, color='lightgreen', alpha=0.5, label = 'worst-case morph scores')
plt.hist(scores_wc_est_max, bins=100, density=True, color='darkgreen', alpha=0.5, label = 'estimated worst-case morph scores (max)')
plt.xlabel('Angle')
plt.xlim(25, 70)
plt.ylabel('frequency')
plt.title('max(Score(morph, id1),Score(morph, id2))')
plt.legend()
plt.savefig(path_eval+'/Angles_wc_max_Histogram_200')
plt.close(fig)  
   
fig=plt.figure()
plt.hist(vgg_wc, bins=50, density=True, color='lightgreen', alpha=0.5, label = 'worst-case morph scores')
plt.hist(vgg_wc_est_max, bins=100, density=True, color='darkgreen', alpha=0.5, label = 'estimated worst-case morph scores (max)')
plt.xlabel('Angle')
plt.xlim(0.45, 1.3)
plt.ylabel('frequency')
plt.title('max(Score(morph, id1),Score(morph, id2))')
plt.legend()
plt.savefig(path_eval+'/VGG_wc_max_Histogram_200')
plt.close(fig)  