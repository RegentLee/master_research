"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import torch.nn as nn
import functools
from .base_model import BaseModel
from . import networks

import torchvision.transforms as transforms

from torchvision.models.resnet import BasicBlock

from util import my_util
from .unet import UNet

from torchinfo import summary

from torch.nn.utils import spectral_norm


class MyPix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='master_pix2pix')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'DP']# 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'Pixel']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # self.net = MyResNet50Generator(opt.input_nc, opt.output_nc, opt.ngf, 
        #                         norm_layer=networks.get_norm_layer(norm_type=opt.norm), 
        #                         use_dropout=(not opt.no_dropout), n_blocks=50)
        self.net = MyUNetGenerator(opt.input_nc, opt.output_nc, opt.ngf, 
                                norm_layer=networks.get_norm_layer(norm_type=opt.norm), 
                                use_dropout=(not opt.no_dropout))
        self.netG = networks.init_net(self.net, opt.init_type, opt.init_gain, self.gpu_ids)
        summary(self.net, input_size=(1, 1, 244, 244), depth=10)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # netD = networks.PixelDiscriminator(opt.output_nc, opt.ndf, norm_layer=networks.get_norm_layer(norm_type=opt.norm))
            netD = MyPixelDiscriminator(opt.input_nc + opt.output_nc)
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
            netPixel = MyPixelDiscriminator(opt.output_nc)
            self.netPixel = networks.init_net(netPixel, opt.init_type, opt.init_gain, self.gpu_ids)
            summary(self.netD, input_size=(1, 2, 244, 244), depth=10)
            summary(self.netPixel, input_size=(1, 1, 244, 244), depth=10)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCE = my_util.CustomCELoss() # torch.nn.CrossEntropyLoss()
            self.criterionL2 = torch.nn.L1Loss()
            self.criterionTri = nn.TripletMarginLoss(margin=50, p=1.0) # torch.nn.MSELoss()# torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Pixel = torch.optim.Adam(self.netPixel.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_Pixel)

            pool = 0
            self.real_c_pool = my_util.ImagePool(pool)
            self.fake_c_pool = my_util.ImagePool(pool)
            self.real_pool = my_util.ImagePool(pool)
            self.fake_pool = my_util.ImagePool(pool)

            self.real_c_pool_G = my_util.ImagePool(pool)
            self.fake_c_pool_G = my_util.ImagePool(pool)
            self.real_pool_G = my_util.ImagePool(pool)
            self.fake_pool_G = my_util.ImagePool(pool)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B.detach()), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # fake_AB = self.fake_B.detach() - self.real_A
        # fake_AB = torch.cat((self.real_A, fake_AB), 1) 
        pred_fake = self.netD(fake_AB.detach())
        # fake1, fake2, fake3, fake4, fake = self.netD(fake_AB.detach())
        ## self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # self.loss_D_fake = self.criterionGAN(pred_fake, False) + self.criterionGAN(fake_all, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = self.real_B - self.real_A
        # real_AB = torch.cat((self.real_A, real_AB), 1)
        pred_real = self.netD(real_AB)
        # real1, real2, real3, real4, real = self.netD(real_AB)
        ## self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D_real = self.criterionGAN(pred_real, True) + self.criterionGAN(real_all, True)
        '''
        """Citation:
        Jolicoeur-Martineau, Alexia. 
        "The relativistic discriminator: a key element missing from standard GAN." 
        arXiv preprint arXiv:1807.00734 (2018).
        """
        '''
        # Fake
        self.loss_D_fake = self.criterionGAN(pred_fake - pred_real.detach(), False)

        # Real
        self.loss_D_real = self.criterionGAN(pred_real - pred_fake.detach(), True)

        # self.loss_D_fake = self.criterionGAN(fake - real, False)
        # self.loss_D_real = self.criterionGAN(real - fake, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # self.loss_D_fake = self.criterionGAN(fake_all - real_all, False)
        # self.loss_D_real = self.criterionGAN(real_all - fake_all, True)
        # self.loss_D += (self.loss_D_fake + self.loss_D_real) * 0.5
        
        # pan_loss = 5*self.criterionL2(real1, fake1)
        # pan_loss += 1.5*self.criterionL2(real2, fake2)
        # pan_loss += 1.5*self.criterionL2(real3, fake3)
        # pan_loss += 1*self.criterionL2(real4, fake4)
        # m = 1
        # print(torch.max(m - pan_loss, torch.tensor([0.]).cuda()))
        # self.loss_D += torch.max(m - pan_loss, torch.tensor([0.]).cuda())[0]

        self.loss_DP = self.loss_D
        self.loss_D.backward()
        '''
        ###########################################################
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        real = self.real_c_pool.query(real_AB)

        fake_AB = torch.cat((self.real_A, self.fake_B.detach()), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        fake = self.fake_c_pool.query(fake_AB)

        # Real > Fake
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake).detach().mean(dim=0, keepdim=True)
        self.loss_D_real = self.criterionGAN(pred_real - pred_fake, True)
        # Fake > True; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real).detach().mean(dim=0, keepdim=True)
        self.loss_D_fake = self.criterionGAN(pred_fake - pred_real, False)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_DP = self.loss_D
        self.loss_D.backward() 
        '''

    def backward_Pixel(self):
        """Calculate GAN loss for the discriminator"""
        
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B.detach()
        pred_fake = self.netPixel(fake_B)
        # self.loss_pixel_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_B = self.real_B
        pred_real = self.netPixel(real_B)
        # self.loss_pixel_real = self.criterionGAN(pred_real, True)

        # Fake
        self.loss_pixel_fake = self.criterionGAN(pred_fake - pred_real.detach(), False)

        # Real
        self.loss_pixel_real = self.criterionGAN(pred_real - pred_fake.detach(), True)
        '''
        """Citation:
        Jolicoeur-Martineau, Alexia. 
        "The relativistic discriminator: a key element missing from standard GAN." 
        arXiv preprint arXiv:1807.00734 (2018).
        """
        '''
        # self.loss_pixel_fake = self.criterionGAN(pred_fake - pred_real, False)
        # elf.loss_pixel_real = self.criterionGAN(pred_real - pred_fake, True)
        self.loss_pixel = (self.loss_pixel_fake + self.loss_pixel_real) * 0.5

        self.loss_DP += self.loss_pixel
        self.loss_pixel.backward()
        '''
        ###########################################################
        real_B = self.real_B
        real = self.real_pool.query(real_B)

        fake_B = self.fake_B.detach()
        fake = self.fake_pool.query(fake_B)

        # pred_fake_1 = self.netPixel(fake.detach())
        # pred_fake_2 = self.netPixel(fake.detach()).mean(dim=0, keepdim=True)
        # print(pred_fake_1, pred_fake_2)

        # Real > Fake
        pred_real = self.netPixel(real_B)
        pred_fake = self.netPixel(fake).detach().mean(dim=0, keepdim=True)
        self.loss_pixel_real = self.criterionGAN(pred_real - pred_fake, True)
        # Fake > True; stop backprop to the generator by detaching fake_B
        pred_fake = self.netPixel(fake_B.detach())
        pred_real = self.netPixel(real).detach().mean(dim=0, keepdim=True)
        self.loss_pixel_fake = self.criterionGAN(pred_fake - pred_real, False)
        
        self.loss_pixel = (self.loss_pixel_fake + self.loss_pixel_real) * 0.5
        self.loss_DP += self.loss_pixel
        self.loss_pixel.backward()
        '''

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # fake_AB = self.fake_B - self.real_A
        # fake_AB = torch.cat((self.real_A, fake_AB), 1)
        pred_fake = self.netD(fake_AB)
        # fake1, fake2, fake3, fake4, fake = self.netD(fake_AB)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = self.real_B - self.real_A
        # real_AB = torch.cat((self.real_A, real_AB), 1)
        # real1, real2, real3, real4, real = self.netD(real_AB)
        pred_real = self.netD(real_AB)
        ## self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True) + self.criterionGAN(fake_all, True)
        self.loss_G_fake = self.criterionGAN(pred_fake - pred_real, True)
        self.loss_G_real = self.criterionGAN(pred_real - pred_fake, False)
        # self.loss_G_fake = self.criterionGAN(fake - real, True)
        # self.loss_G_real = self.criterionGAN(real - fake, False)
        self.loss_G_GAN = (self.loss_G_fake + self.loss_G_real) * 0.5
        
        # Fake
        fake_B = self.fake_B
        pred_fake = self.netPixel(fake_B)
        # self.loss_G_GAN += self.criterionGAN(pred_fake, True)
        # Real
        real_B = self.real_B
        pred_real = self.netPixel(real_B)
        # self.loss_G_GAN += self.criterionGAN(pred_fake, True)
        self.loss_G_fake = self.criterionGAN(pred_fake - pred_real, True)
        self.loss_G_real = self.criterionGAN(pred_real - pred_fake, False)
        self.loss_G_GAN += (self.loss_G_fake + self.loss_G_real) * 0.5
        

        # pan_loss = 5*self.criterionL2(real1, fake1)
        # pan_loss += 1.5*self.criterionL2(real2, fake2)
        # pan_loss += 1.5*self.criterionL2(real3, fake3)
        # pan_loss += 1*self.criterionL2(real4, fake4)
        
        # self.loss_G_GAN = self.loss_G_fake
        # Second, G(A) = B
        """A = ((self.real_A + 1)*my_util.distance[-1]).to(torch.double)
        m_max = my_util.distance[-1]
        m_min = 0
        A = torch.where(A > m_max, m_max, A)
        A = ((A - m_min)/(m_max - m_min)*2 - 1).to(torch.float)"""
        
        self.loss_G_L1 = self.criterionL2(self.fake_B, self.real_B)
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real = self.netD(real_AB)
        # self.loss_G_L1 = self.criterionTri(self.fake_B, self.real_B, self.real_A)
        # self.loss_G_L1 = torch.max(self.criterionL2(self.fake_B, self.real_B) - self.criterionTri(self.fake_B, A) + 0.01)
        # self.criterionTri(self.fake_B, self.real_B, A)
        
        self.loss_G_L1 = self.loss_G_L1 * self.opt.lambda_L1
        # self.loss_G_L1 += self.criterionTri(self.fake_B, self.real_B)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # self.loss_G = pan_loss + self.loss_G_GAN
        self.loss_G.backward()
        '''

        ###########################################################
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        real = self.real_c_pool_G.query(real_AB)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake = self.fake_c_pool_G.query(fake_AB.detach())
        if len(fake) != 1:
            fake = torch.split(fake, len(fake) - 1, dim=0)[0]
            fake = torch.cat([fake, fake_AB], dim=0)
        else:
            fake = torch.cat((self.real_A, self.fake_B), 1)

        # Real > Fake
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB).mean(dim=0, keepdim=True)
        self.loss_G_real = self.criterionGAN(pred_fake - pred_real, True)
        # Fake > True; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake_AB)
        pred_real = self.netD(real_AB).mean(dim=0, keepdim=True)
        self.loss_G_fake = self.criterionGAN(pred_real - pred_fake, False)

        self.loss_G_GAN = (self.loss_G_fake + self.loss_G_real) * 0.5
        
        ###########################################################
        real_B = self.real_B
        real = self.real_pool_G.query(real_B)

        fake_B = self.fake_B
        fake = self.fake_pool_G.query(fake_B.detach())
        if len(fake) != 1:
            fake = torch.split(fake, len(fake) - 1, dim=0)[0]
            fake = torch.cat([fake, fake_B], dim=0)
        else:
            fake = self.fake_B

        # Real > Fake
        pred_real = self.netPixel(real_B)
        pred_fake = self.netPixel(fake_B).mean(dim=0, keepdim=True)
        self.loss_pixel_real = self.criterionGAN(pred_fake - pred_real, True)
        # Fake > True; stop backprop to the generator by detaching fake_B
        pred_fake = self.netPixel(fake_B)
        pred_real = self.netPixel(real_B)# .mean(dim=0, keepdim=True)
        self.loss_pixel_fake = self.criterionGAN(pred_real - pred_fake, False)
        
        self.loss_G_GAN += (self.loss_pixel_fake + self.loss_pixel_real) * 0.5
        
        self.loss_G = self.loss_G_GAN
        self.loss_G.backward()
        '''

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        for _ in range(1):
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        
        # update Pixel
        for _ in range(1):
            self.set_requires_grad(self.netPixel, True)  # enable backprop for Pixel
            self.optimizer_Pixel.zero_grad()     # set Pixel's gradients to zero
            self.backward_Pixel()                # calculate gradients for Pixel
            self.optimizer_Pixel.step()          # update Pixel's weights
        
        # update G
        for _ in range(1):
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.set_requires_grad(self.netPixel, False) 
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights

        # print("Model's state_dict:")
        # print(self.net.state_dict()["linear.weight"])
        # print(self.net.state_dict()["linear.bias"])

class Synchro(nn.Module):
    def __init__(self):
        super(Synchro, self).__init__()

    def make_input(self, input):
        ip0 = input[:, :, 0:64, 0:64]
        ip1 = input[:, :, 0:64, 60:124]
        ip2 = input[:, :, 0:64, 120:184]
        ip3 = input[:, :, 0:64, 180:244]

        ip4 = input[:, :, 60:124, 0:64]
        ip5 = input[:, :, 60:124, 60:124]
        ip6 = input[:, :, 60:124, 120:184]
        ip7 = input[:, :, 60:124, 180:244]

        ip8 = input[:, :, 120:184, 0:64]
        ip9 = input[:, :, 120:184, 60:124]
        ip10 = input[:, :, 120:184, 120:184]
        ip11 = input[:, :, 120:184, 180:244]

        ip12 = input[:, :, 180:244, 0:64]
        ip13 = input[:, :, 180:244, 60:124]
        ip14 = input[:, :, 180:244, 120:184]
        ip15 = input[:, :, 180:244, 180:244]

        ip = [
            ip0, ip1, ip2, ip3,
            ip4, ip5, ip6, ip7,
            ip8, ip9, ip10, ip11,
            ip12, ip13, ip14, ip15
        ]

        ip = torch.cat(ip, dim=1)

        return ip

    def make_output(self, input):
        output = torch.empty((input.shape[0], 1, 244, 244), device=input.device)

        output[:, :, 0:60, 0:60] = input[:, 0, 0:60, 0:60]
        output[:, :, 0:60, 64:120] = input[:, 1, 0:60, 4:60]
        output[:, :, 0:60, 124:180] = input[:, 2, 0:60, 4:60]
        output[:, :, 0:60, 184:244] = input[:, 3, 0:60, 4:64]

        output[:, :, 64:120, 0:60] = input[:, 4, 4:60, 0:60]
        output[:, :, 64:120, 64:120] = input[:, 5, 4:60, 4:60]
        output[:, :, 64:120, 124:180] = input[:, 6, 4:60, 4:60]
        output[:, :, 64:120, 184:244] = input[:, 7, 4:60, 4:64]

        output[:, :, 124:180, 0:60] = input[:, 8, 4:60, 0:60]
        output[:, :, 124:180, 64:120] = input[:, 9, 4:60, 4:60]
        output[:, :, 124:180, 124:180] = input[:, 10, 4:60, 4:60]
        output[:, :, 124:180, 184:244] = input[:, 11, 4:60, 4:64]

        output[:, :, 184:244, 0:60] = input[:, 12, 4:64, 0:60]
        output[:, :, 184:244, 64:120] = input[:, 13, 4:64, 4:60]
        output[:, :, 184:244, 124:180] = input[:, 14, 4:64, 4:60]
        output[:, :, 184:244, 184:244] = input[:, 15, 4:64, 4:64]

        output[:, :, 0:60, 60:64] = (input[:, 0, 0:60, 60:64] + input[:, 1, 0:60, 0:4])/2
        output[:, :, 0:60, 120:124] = (input[:, 1, 0:60, 60:64] + input[:, 2, 0:60, 0:4])/2
        output[:, :, 0:60, 180:184] = (input[:, 2, 0:60, 60:64] + input[:, 3, 0:60, 0:4])/2

        output[:, :, 64:120, 60:64] = (input[:, 4, 4:60, 60:64] + input[:, 5, 4:60, 0:4])/2
        output[:, :, 64:120, 120:124] = (input[:, 5, 4:60, 60:64] + input[:, 6, 4:60, 0:4])/2
        output[:, :, 64:120, 180:184] = (input[:, 6, 4:60, 60:64] + input[:, 7, 4:60, 0:4])/2

        output[:, :, 124:180, 60:64] = (input[:, 8, 4:60, 60:64] + input[:, 9, 4:60, 0:4])/2
        output[:, :, 124:180, 120:124] = (input[:, 9, 4:60, 60:64] + input[:, 10, 4:60, 0:4])/2
        output[:, :, 124:180, 180:184] = (input[:, 10, 4:60, 60:64] + input[:, 11, 4:60, 0:4])/2

        output[:, :, 184:244, 60:64] = (input[:, 12, 4:64, 60:64] + input[:, 13, 4:64, 0:4])/2
        output[:, :, 184:244, 120:124] = (input[:, 13, 4:64, 60:64] + input[:, 14, 4:64, 0:4])/2
        output[:, :, 184:244, 180:184] = (input[:, 14, 4:64, 60:64] + input[:, 15, 4:64, 0:4])/2

        output[:, :, 60:64, 0:60] = (input[:, 0, 60:64, 0:60] + input[:, 4, 0:4, 0:60])/2
        output[:, :, 120:124, 0:60] = (input[:, 4, 60:64, 0:60] + input[:, 8, 0:4, 0:60])/2
        output[:, :, 180:184, 0:60] = (input[:, 8, 60:64, 0:60] + input[:, 12, 0:4, 0:60])/2

        output[:, :, 60:64, 64:120] = (input[:, 1, 60:64, 4:60] + input[:, 5, 0:4, 4:60])/2
        output[:, :, 120:124, 64:120] = (input[:, 5, 60:64, 4:60] + input[:, 9, 0:4, 4:60])/2
        output[:, :, 180:184, 64:120] = (input[:, 9, 60:64, 4:60] + input[:, 13, 0:4, 4:60])/2

        output[:, :, 60:64, 124:180] = (input[:, 2, 60:64, 4:60] + input[:, 6, 0:4, 4:60])/2
        output[:, :, 120:124, 124:180] = (input[:, 6, 60:64, 4:60] + input[:, 10, 0:4, 4:60])/2
        output[:, :, 180:184, 124:180] = (input[:, 10, 60:64, 4:60] + input[:, 14, 0:4, 4:60])/2

        output[:, :, 60:64, 184:244] = (input[:, 3, 60:64, 4:64] + input[:, 7, 0:4, 4:64])/2
        output[:, :, 120:124, 184:244] = (input[:, 7, 60:64, 4:64] + input[:, 11, 0:4, 4:64])/2
        output[:, :, 180:184, 184:244] = (input[:, 11, 60:64, 4:64] + input[:, 15, 0:4, 4:64])/2

        output[:, :, 60:64, 60:64] = (input[:, 0, 60:64, 60:64] + input[:, 1, 60:64, 0:4] + input[:, 4, 0:4, 60:64] + input[:, 5, 0:4, 0:4])/4
        output[:, :, 120:124, 60:64] = (input[:, 4, 60:64, 60:64] + input[:, 5, 60:64, 0:4] + input[:, 8, 0:4, 60:64] + input[:, 9, 0:4, 0:4])/4
        output[:, :, 180:184, 60:64] = (input[:, 8, 60:64, 60:64] + input[:, 9, 60:64, 0:4] + input[:, 12, 0:4, 60:64] + input[:, 13, 0:4, 0:4])/4

        output[:, :, 60:64, 120:124] = (input[:, 1, 60:64, 60:64] + input[:, 2, 60:64, 0:4] + input[:, 5, 0:4, 60:64] + input[:, 6, 0:4, 0:4])/4
        output[:, :, 120:124, 120:124] = (input[:, 5, 60:64, 60:64] + input[:, 6, 60:64, 0:4] + input[:, 9, 0:4, 60:64] + input[:, 10, 0:4, 0:4])/4
        output[:, :, 180:184, 120:124] = (input[:, 9, 60:64, 60:64] + input[:, 10, 60:64, 0:4] + input[:, 13, 0:4, 60:64] + input[:, 14, 0:4, 0:4])/4

        output[:, :, 60:64, 180:184] = (input[:, 2, 60:64, 60:64] + input[:, 3, 60:64, 0:4] + input[:, 6, 0:4, 60:64] + input[:, 7, 0:4, 0:4])/4
        output[:, :, 120:124, 180:184] = (input[:, 6, 60:64, 60:64] + input[:, 7, 60:64, 0:4] + input[:, 10, 0:4, 60:64] + input[:, 11, 0:4, 0:4])/4
        output[:, :, 180:184, 180:184] = (input[:, 10, 60:64, 60:64] + input[:, 11, 60:64, 0:4] + input[:, 14, 0:4, 60:64] + input[:, 15, 0:4, 0:4])/4

        return output

class MyUNetGenerator(Synchro):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
        """
        super(MyUNetGenerator, self).__init__()

        model = [UNet(input_nc, output_nc, True)]

        model += [nn.Tanh()]
        # model += [nn.Hardtanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        input = self.make_input(input)
        x = self.model(input)
        x = self.make_output(x)

        # x_t = x.transpose(2, 3)
        # x = (x + x_t)/2

        return x       

class MyUNetD(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
        """
        super(MyUNetD, self).__init__()

        model = [UNetD(input_nc, 1, True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        x, tf = self.model(input)

        return x, tf

'''
class MyUNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
        """
        super(MyUNetGenerator, self).__init__()

        model = [UNet(input_nc, 64, False)]
        
        model += [nn.Flatten(start_dim=2)]

        self.model = nn.Sequential(*model)

        self.softmax = nn.Softmax(dim=2)
        self.maxpool = nn.MaxPool1d(64, return_indices=True)
        self.unpool = nn.MaxUnpool1d(64)
        self.linear = nn.Linear(64, 1)

    def forward(self, input):
        """Standard forward"""
        x_CE = self.model(input).transpose(1, 2)
        m = self.softmax(x_CE)
        prob, i = self.maxpool(m)
        m = self.unpool(prob, i, output_size=x_CE.size())
        m = self.linear(m)*1.4
        m_max = my_util.distance[-1] # torch.max(torch.flatten(m))
        m_min = 0 # torch.min(torch.flatten(m))
        fake_image = (torch.div(m - m_min, m_max - m_min)*2 - 1).view(-1, 1, 64, 64)

        image = torch.zeros_like(i, dtype=torch.float)
        image = (2 + (i - 1)*0.32).to(torch.float)
        image = torch.where(image < 2, torch.Tensor([0.]).cuda(), image)
        m_max = my_util.distance[-1] # torch.max(torch.flatten(m))
        m_min = 0 # torch.min(torch.flatten(m))
        image = (torch.div(image - m_min, m_max - m_min)*2 - 1).view(-1, 1, 64, 64)

        return fake_image, x_CE, image, prob.view(-1, 1, 64, 64)
'''

class MyResnet50Generator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(MyResnet50Generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        model += [Bottleneck(64, 256, True),
                  Bottleneck(256, 256),
                  Bottleneck(256, 256),
                  
                  Bottleneck(256, 512),
                  Bottleneck(512, 512),
                  Bottleneck(512, 512),
                  Bottleneck(512, 512),

                  Bottleneck(512, 1024)]

        for i in range(22):
            model += [Bottleneck(1024, 1024)]
        
        model += [Bottleneck(1024, 2048),
                  Bottleneck(2048, 2048),
                  Bottleneck(2048, 2048)]

        model += [nn.Conv2d(2048, 64*64, kernel_size=1), norm_layer(64*64), nn.ReLU(True)]
        model += [nn.Flatten(start_dim=2)]
        model += [nn.Linear(64, 64)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        # self.argmax = nn.MaxPool1d(64)#, return_indices=True)

        self.softmax = nn.Softmax(dim=2)
        self.maxpool = nn.MaxPool1d(64, return_indices=True)
        self.unpool = nn.MaxUnpool1d(64)
        self.linear = nn.Linear(64, 1)

    def forward(self, input):
        """Standard forward"""
        x_CE = self.model(input).view(-1, 4096, 64)
        m = self.softmax(x_CE)
        prob, i = self.maxpool(m)
        # m = self.unpool(prob, i, output_size=x_CE.size())
        m = self.linear(m)
        m_max = my_util.distance[-1] # torch.max(torch.flatten(m))
        m_min = 0 # torch.min(torch.flatten(m))
        fake_image = (torch.div(m - m_min, m_max - m_min)*2 - 1).view(-1, 1, 64, 64)

        image = torch.zeros_like(i, dtype=torch.float)
        image = (2 + (i - 1)*0.32).to(torch.float)
        image = torch.where(image < 2, torch.Tensor([0.]).cuda(), image)
        m_max = my_util.distance[-1] # torch.max(torch.flatten(m))
        m_min = 0 # torch.min(torch.flatten(m))
        image = (torch.div(image - m_min, m_max - m_min)*2 - 1).view(-1, 1, 64, 64)

        return fake_image, x_CE, image, prob.view(-1, 1, 64, 64)

class ResNet(nn.Module):
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d): 
          
        super(ResNet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d   

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=1),
            BasicBlock(64, 64, norm_layer=norm_layer),
            BasicBlock(64, 64, norm_layer=norm_layer),
            # BasicBlock(64, 64, norm_layer=norm_layer),

            BasicBlock(64, 128, 2, downsample=nn.Conv2d(64, 128, 1, stride=2), norm_layer=norm_layer),
            BasicBlock(128, 128, norm_layer=norm_layer),
            # BasicBlock(128, 128, norm_layer=norm_layer),
            # BasicBlock(128, 128, norm_layer=norm_layer),

            BasicBlock(128, 256, 2, downsample=nn.Conv2d(128, 256, 1, stride=2), norm_layer=norm_layer),
            BasicBlock(256, 256, norm_layer=norm_layer),
            # BasicBlock(256, 256, norm_layer=norm_layer),
            # BasicBlock(256, 256, norm_layer=norm_layer),
            # BasicBlock(256, 256, norm_layer=norm_layer),
            # BasicBlock(256, 256, norm_layer=norm_layer),

            BasicBlock(256, 512, 2, downsample=nn.Conv2d(256, 512, 1, stride=2), norm_layer=norm_layer),
            BasicBlock(512, 512, norm_layer=norm_layer),
            # BasicBlock(512, 512, norm_layer=norm_layer),

            nn.AdaptiveAvgPool2d((1, 1))     
        )
        
        # Postreior Block
        # self.output = nn.Conv2d(2048, 1, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))        
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model(x)
        
        # Postreior Block
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet50(nn.Module):
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d): 
          
        super(ResNet50, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ndf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ndf),
                 nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        
        # Residual blocks
        self.resblock1 = Bottleneck(64, 256, True)
        self.resblock2 = Bottleneck(256, 256)
        self.resblock3 = Bottleneck(256, 256)
        self.resblock4 = Bottleneck(256, 512)
        self.resblock5 = Bottleneck(512, 512)
        self.resblock6 = Bottleneck(512, 512)
        self.resblock7 = Bottleneck(512, 512)
        self.resblock8 = Bottleneck(512, 1024)
        self.resblock9 = Bottleneck(1024, 1024)
        self.resblock10 =Bottleneck(1024, 1024)
        self.resblock11 =Bottleneck(1024, 1024)
        self.resblock12 =Bottleneck(1024, 1024)
        self.resblock13 =Bottleneck(1024, 1024)
        self.resblock14 =Bottleneck(1024, 2048)
        self.resblock15 =Bottleneck(2048, 2048)
        self.resblock16 =Bottleneck(2048, 2048)
        
        # Postreior Block
        # self.output = nn.Conv2d(2048, 1, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))        
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.model(x)
        
        # Residual blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.resblock16(x)
        
        # Postreior Block
        # x = self.output(x)
        x = self.glob_avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    """
    Bottleneckを使用したresidual blockクラス
    """
    def __init__(self, indim, outdim, is_first_resblock=False):
        super(Bottleneck, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # W, Hを小さくしてCを増やす際はstrideを2にする +
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            if is_first_resblock:
                # 最初のresblockは(W､ H)は変更しないのでstrideは1にする
                stride = 1
            else:
                stride = 2
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=stride)
        else:
            stride = 1
        
        dim_inter = int(outdim / 4)
        self.conv1 = nn.Conv2d(indim, dim_inter , 1)
        self.bn1 = nn.BatchNorm2d(dim_inter)
        self.conv2 = nn.Conv2d(dim_inter, dim_inter, 3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(dim_inter)
        self.conv3 = nn.Conv2d(dim_inter, outdim, 1)
        self.bn3 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out


class MyResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(MyResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [networks.ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, 64, kernel_size=7, padding=0), norm_layer(64), nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=1)]

        # model += [nn.Conv2d(ngf * mult, 64*64, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(64*64), nn.ReLU(True)]
        
        model += [nn.Flatten(start_dim=2)]
        # model += [nn.Linear(64, 64)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        # self.argmax = nn.MaxPool1d(64)#, return_indices=True)

        self.softmax = nn.Softmax(dim=2)
        self.maxpool = nn.MaxPool1d(64, return_indices=True)
        self.unpool = nn.MaxUnpool1d(64)
        self.linear = nn.Linear(64, 1)

    def forward(self, input):
        """Standard forward"""
        x_CE = self.model(input).view(-1, 4096, 64)
        m = self.softmax(x_CE)
        prob, i = self.maxpool(m)
        # m = self.unpool(prob, i, output_size=x_CE.size())
        m = self.linear(m)
        m_max = my_util.distance[-1] # torch.max(torch.flatten(m))
        m_min = 0 # torch.min(torch.flatten(m))
        fake_image = (torch.div(m - m_min, m_max - m_min)*2 - 1).view(-1, 1, 64, 64)

        image = torch.zeros_like(i, dtype=torch.float)
        image = (2 + (i - 1)*0.32).to(torch.float)
        image = torch.where(image < 2, torch.Tensor([0.]).cuda(), image)
        m_max = my_util.distance[-1] # torch.max(torch.flatten(m))
        m_min = 0 # torch.min(torch.flatten(m))
        image = (torch.div(image - m_min, m_max - m_min)*2 - 1).view(-1, 1, 64, 64)

        return fake_image, x_CE, image, prob.view(-1, 1, 64, 64)


class MyD(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(MyD, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),
            # nn.Conv2d(512, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        '''self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=4),
            nn.PReLU(),
            norm_layer(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=4),
            nn.PReLU(),
            norm_layer(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.PReLU(),
            norm_layer(256),
            nn.Conv2d(256, 1, kernel_size=1),
        )'''

    def forward(self, input):
        """Standard forward."""
        x = self.model(input)
        return x

def focal_loss(fake, real):
    epsilon = 1e-8

    alpha = 1

    ## 2D DFT with orthonomalization
    fake_fft = torch.fft.fft2(fake, norm = 'ortho')

    real_fft = torch.fft.fft2(real, norm = 'ortho')

    x_dist = (real_fft.real - fake_fft.real) ** 2

    y_dist = (real_fft.imag - fake_fft.imag) ** 2

    distance_matrix = torch.sqrt(x_dist + y_dist + epsilon) 

    ## squared Eucliedean distance
    squared_distance = distance_matrix ** 2

    ## weight for spatial frequency
    weight_matrix = distance_matrix ** alpha

    # normalization weight_matrix to [0,1]
    norm_weight_matrix = (weight_matrix - torch.min(weight_matrix)) /  (torch.max(weight_matrix) - torch.min(weight_matrix))


    prod = torch.mul(squared_distance, norm_weight_matrix)

    FFL = torch.sum(prod) / (64 * 64 * 1)

    return FFL


class PAN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PAN, self).__init__()

        self.output1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        self.output2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )

        self.output3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.output4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(512, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, 1, kernel_size=1)
        )
    
    def forward(self, x):
        output1 = self.output1(x)
        output2 = self.output2(output1)
        output3 = self.output3(output2)
        output4 = self.output4(output3)
        output = self.output(output4)

        return output1, output2, output3, output4, output


class MyPixelDiscriminator(Synchro):
    def __init__(self, input_nc):
        super(MyPixelDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d
        ndf = 64
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), #(1, 256, 256) -> (64, 128, 128) # (1, 64, 64) -> (64, 32, 32)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1), #(64, 128, 128) -> (128, 64, 64) # (64, 32, 32) -> (128, 16, 16)
            norm_layer(ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1), #(128, 64, 64) -> (256, 32, 32) # (128, 16, 16) -> (256, 8, 8)
            norm_layer(ndf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1), #(256, 32, 32) -> (512, 16, 16) # (256, 8, 8) -> (512, 4, 4)
            norm_layer(ndf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, padding=0), # (512, 4, 4) -> (1, 1, 1)
        )
        '''
            nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1), #(512, 16, 16) -> (1024, 8, 8) # (512, 4, 4) -> (1, 1, 1)
            norm_layer(ndf*16),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*16, ndf*32, kernel_size=4, stride=2, padding=1), #(1024, 8, 8) -> (2048, 4, 4)
            norm_layer(ndf*32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*32, 1, kernel_size=3, stride=3, padding=0) #(2048, 4, 4) -> (1, 1, 1)   
        )'''

    def forward(self, x):
        x = self.make_input(x)
        return self.model(x).view(-1, 1)

class NoNorm(nn.Module):
    def __init__(self, dim):
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

class LNorm(nn.Module):
    def __init__(self, dim):
        super(LNorm, self).__init__()

    def forward(self, x):
        x = nn.functional.layer_norm(x, x.size()[1:])
        return x
