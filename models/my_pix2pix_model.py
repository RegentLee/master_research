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
from .unet import UNet, IResNet

from torchinfo import summary

from torch.nn.utils import spectral_norm

import numpy as np


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
        parser.set_defaults(input_nc=2, output_nc=1)  # specify dataset-specific default values
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
        self.loss_names = ['G_GAN', 'DP', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['a', 'fake_B', 'b', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'Pixel']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # self.net = MyResNet50Generator(opt.input_nc, opt.output_nc, opt.ngf, 
        #                         norm_layer=networks.get_norm_layer(norm_type=opt.norm), 
        #                         use_dropout=(not opt.no_dropout), n_blocks=50)
        # self.net = MyUNetGenerator(opt.input_nc, opt.output_nc, opt.ngf, 
        #                         norm_layer=networks.get_norm_layer(norm_type=opt.norm), 
        #                         use_dropout=(not opt.no_dropout))
        self.net = Generator(opt.input_nc, opt.output_nc)
        # self.net = networks.UnetGenerator(opt.input_nc, opt.output_nc, 6, opt.ngf, norm_layer=networks.get_norm_layer(norm_type=opt.norm), use_dropout=(not opt.no_dropout))
        self.netG = networks.init_net(self.net, opt.init_type, opt.init_gain, self.gpu_ids)
        summary(self.net, input_data=[torch.zeros([1, 2, 240, 240]), torch.zeros([1, 2])], depth=15) #, torch.zeros(1, dtype=torch.long)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # netD = networks.PixelDiscriminator(opt.output_nc, opt.ndf, norm_layer=networks.get_norm_layer(norm_type=opt.norm))
            netD = MyPixelDiscriminator(opt.input_nc + opt.output_nc, nn.Identity)
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
            netPixel = MyPixelDiscriminator(opt.output_nc, nn.Identity)
            self.netPixel = networks.init_net(netPixel, opt.init_type, opt.init_gain, self.gpu_ids)
            summary(self.netD, input_data=[torch.zeros([1, 3, 263, 263]), torch.zeros([1, 1], dtype=torch.long)], depth=10)
            summary(self.netPixel, input_data=[torch.zeros([1, 1, 240, 240]), torch.zeros([1, 1], dtype=torch.long)], depth=10)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionAB = torch.nn.BCEWithLogitsLoss()
            self.criterionTri = nn.TripletMarginLoss(margin=1, p=1.0) # torch.nn.MSELoss()# torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*4, betas=(opt.beta1, 0.999))
            self.optimizer_Pixel = torch.optim.Adam(self.netPixel.parameters(), lr=opt.lr*4, betas=(opt.beta1, 0.999))
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

            # DA
            self.rotate = Rotate()
            self.cutout = Cutout()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'] # .to(self.device)
        self.a = self.real_A
        self.Pos = input['Pos']
        self.real_A = torch.cat([self.real_A, self.Pos], dim=1).to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.real_B = self._shortcut(self.idt_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.idx = input['idx'].to(self.device)
        self.a = self._shortcut(self.a)
        
        self.domain = input['domain'].to(self.device)
        self.domain = self._to_onehot(self.domain)

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
        return x

    def _to_onehot(self, idx):
        onehot = []
        for i in range(len(idx)):
            if idx[i] == 0:
                onehot.append([1, 0])
            elif idx[i] == 1:
                onehot.append([0, 1])
        onehot = torch.tensor(onehot)
        return onehot

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A, self.domain)  # G(A)
        x = self.fake_B
        x = (x + 1)*my_util.distance[-1]//2
        b = (self.real_B + 1)*my_util.distance[-1]//2
        # m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        # m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = x - b
        m_max = torch.max(x)
        m_min = torch.min(x)
        self.b = (x - m_min)/(m_max - m_min)*2 - 1

    def backward_D(self):
        # real_A, fake_B, real_B = self.rotate(self.real_A.clone(), self.fake_B.detach().clone(), self.real_B.clone())
        # real_A, fake_B, real_B = self.cutout(self.real_A.clone(), self.fake_B.detach().clone(), self.real_B.clone())
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B.detach()), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # fake_AB = self.fake_B.detach()
        pred_fake = self.netD(fake_AB.detach(), self.idx)
        # fake1, fake2, fake3, fake4, fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # self.loss_D_fake = self.criterionGAN(pred_fake, False) + self.criterionGAN(fake_all, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = self.real_B
        # real_AB = torch.cat((self.real_A, real_AB), 1)
        pred_real = self.netD(real_AB, self.idx)
        # real1, real2, real3, real4, real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D_real = self.criterionGAN(pred_real, True) + self.criterionGAN(real_all, True)

        # gp, _ = networks.cal_gradient_penalty(self.netD, real_AB, fake_AB, self.device, self.idx)
        '''
        """Citation:
        Jolicoeur-Martineau, Alexia. 
        "The relativistic discriminator: a key element missing from standard GAN." 
        arXiv preprint arXiv:1807.00734 (2018).
        """
        '''
        # Fake
        ## self.loss_D_fake = self.criterionGAN(pred_fake - pred_real.detach(), False)

        # Real
        ## self.loss_D_real = self.criterionGAN(pred_real - pred_fake.detach(), True)

        # self.loss_D_fake = self.criterionGAN(fake - real, False)
        # self.loss_D_real = self.criterionGAN(real - fake, True)
        # combine loss and calculate gradients
        self.loss_D = self.loss_D_fake + self.loss_D_real#  + gp

        if self.opt.gan_mode == 'wgangp':
            gp, _ = networks.cal_gradient_penalty(self.netD, real_AB, fake_AB, self.device, self.idx)
            self.loss_D += gp

        '''
        fake_AB_r = torch.rot90(fake_AB, 2, [2, 3])
        pred_fake_r = self.netD(fake_AB_r.detach(), self.idx)
        pred_fake_r = torch.rot90(pred_fake_r, 2, [2, 3])
        self.loss_D += 10*self.criterionL2(pred_fake, pred_fake_r)
        # Real
        real_AB_r = torch.rot90(real_AB, 2, [2, 3])
        pred_real_r = self.netD(real_AB_r, self.idx)
        pred_real_r = torch.rot90(pred_real_r, 2, [2, 3])
        self.loss_D += 10*self.criterionL2(pred_real, pred_real_r)
        '''
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

        # self.loss_D += (self.criterionCE(ce_fake, self.idx) + self.criterionCE(ce_real, self.idx)) * 0.5

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
        pred_fake = self.netPixel(fake_B, self.idx)
        self.loss_pixel_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_B = self.real_B
        pred_real = self.netPixel(real_B, self.idx)
        self.loss_pixel_real = self.criterionGAN(pred_real, True)

        # gp, _ = networks.cal_gradient_penalty(self.netPixel, real_B, fake_B, self.device, self.idx)

        # Fake
        # self.loss_pixel_fake = self.criterionGAN(pred_fake - pred_real.detach(), False)

        # Real
        # self.loss_pixel_real = self.criterionGAN(pred_real - pred_fake.detach(), True)
        '''
        """Citation:
        Jolicoeur-Martineau, Alexia. 
        "The relativistic discriminator: a key element missing from standard GAN." 
        arXiv preprint arXiv:1807.00734 (2018).
        """
        '''
        # self.loss_pixel_fake = self.criterionGAN(pred_fake - pred_real, False)
        # elf.loss_pixel_real = self.criterionGAN(pred_real - pred_fake, True)
        self.loss_pixel = self.loss_pixel_fake + self.loss_pixel_real#  + gp
        if self.opt.gan_mode == 'wgangp':
            gp, _ = networks.cal_gradient_penalty(self.netPixel, real_B, fake_B, self.device, self.idx)
            self.loss_pixel += gp

        # self.loss_pixel += (self.criterionCE(ce_fake, self.idx) + self.criterionCE(ce_real, self.idx)) * 0.5

        # _, _, r_a = self.netPixel(self.a)
        # self.loss_pixel += (self.criterionAB(f_b, torch.ones_like(f_b)) + self.criterionAB(r_b, torch.ones_like(r_b)) + self.criterionAB(r_a, torch.zeros_like(r_a))) * 0.33

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
        # fake_AB = self.fake_B
        # fake_AB = torch.cat((self.real_A, fake_AB), 1)
        pred_fake = self.netD(fake_AB, self.idx)
        # fake1, fake2, fake3, fake4, fake = self.netD(fake_AB)
        # Real
        ## real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = self.real_B - self.real_A
        # real_AB = torch.cat((self.real_A, real_AB), 1)
        # real1, real2, real3, real4, real = self.netD(real_AB)
        ## pred_real = self.netD(real_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True) + self.criterionGAN(fake_all, True)
        ## self.loss_G_fake = self.criterionGAN(pred_fake - pred_real, True, True)
        ## self.loss_G_real = self.criterionGAN(pred_real - pred_fake, False, True)
        # self.loss_G_fake = self.criterionGAN(fake - real, True)
        # self.loss_G_real = self.criterionGAN(real - fake, False)
        ## self.loss_G_GAN = (self.loss_G_fake + self.loss_G_real) * 0.5
        
        '''
        # Fake
        fake_B = self.fake_B
        pred_fake = self.netPixel(fake_B, self.idx)
        self.loss_G_GAN += self.criterionGAN(pred_fake, True, for_discriminator=False)
        '''
        '''
        # Real
        real_B = self.real_B
        pred_real = self.netPixel(real_B)
        # self.loss_G_GAN += self.criterionGAN(pred_fake, True)
        self.loss_G_fake = self.criterionGAN(pred_fake - pred_real, True, True)
        self.loss_G_real = self.criterionGAN(pred_real - pred_fake, False, True)
        self.loss_G_GAN += (self.loss_G_fake + self.loss_G_real) * 0.5
        '''
        # _, _, r_a = self.netPixel(self.a)
        # self.loss_G_GAN += (self.criterionAB(f_b, torch.ones_like(f_b)) + self.criterionAB(r_b, torch.ones_like(r_b)) + self.criterionAB(r_a, torch.zeros_like(r_a))) * 0.33
        

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
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # self.loss_G_L1 = self.criterionTri(self.fake_B, self.real_B, self.a)
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real = self.netD(real_AB)
        # self.loss_G_L1 = self.criterionTri(self.fake_B, self.real_B, self.real_A)
        # self.loss_G_L1 = torch.max(self.criterionL2(self.fake_B, self.real_B) - self.criterionTri(self.fake_B, A) + 0.01)
        # self.criterionTri(self.fake_B, self.real_B, A)
        
        self.loss_G_L1 = self.loss_G_L1 * self.opt.lambda_L1
        # self.loss_G_L1 += self.criterionTri(self.fake_B, self.real_B)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # self.idt_B = self.netG(self.real_B)
        # self.loss_idt = self.criterionL2(self.idt_B, self.real_B) * 5
        # self.loss_G += self.loss_idt

        # self.loss_G += (self.criterionCE(ce_fake, self.idx) + self.criterionCE(ce_real, self.idx)) * 0.5

        # idt4B = self.netG(self.idt_B)
        # self.loss_G += self.criterionL2(idt4B, self.real_B) * 0.33

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
        '''
        # update Pixel
        for _ in range(1):
            self.set_requires_grad(self.netPixel, True)  # enable backprop for Pixel
            self.optimizer_Pixel.zero_grad()     # set Pixel's gradients to zero
            self.backward_Pixel()                # calculate gradients for Pixel
            self.optimizer_Pixel.step()          # update Pixel's weights
        '''
        # update G
        for i in range(1):
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.set_requires_grad(self.netPixel, False) 
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
            # if i == 0:
            #     self.forward()   

        # print("Model's state_dict:")
        # print(self.net.state_dict()["linear.weight"])
        # print(self.net.state_dict()["linear.bias"])

class Rotate:
    def __init__(self, p=0.5) -> None:
        self.prob = p

    def __call__(self, real_A, fake_B, real_B):
        p = np.random.rand()
        if p < self.prob:
            real_A = torch.rot90(real_A, 2, [2, 3])
            fake_B = torch.rot90(fake_B, 2, [2, 3])
            real_B = torch.rot90(real_B, 2, [2, 3])
        return real_A, fake_B, real_B

class Cutout:
    def __init__(self, p=0.5, cutout_range=10) -> None:
        self.prob = p
        self.cr = cutout_range

    def __call__(self, real_A, fake_B, real_B):
        p = np.random.rand()
        if p < self.prob:
            n = np.random.randint(0, self.cr + 1)

            for _ in range(n):
                len_A = len(real_A[0][0])
                c = np.random.randint(1, len_A - 1)

                real_A[:, :, c] = 0
                real_A[:, :, :, c] = 0

                fake_B[:, :, c] = 0
                fake_B[:, :, :, c] = 0

                real_B[:, :, c] = 0
                real_B[:, :, :, c] = 0

        return real_A, fake_B, real_B

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

        # self.PE = PositionEmbedding()

        model = [UNet(input_nc, output_nc, True)]

        model += [nn.Tanh()]
        # model += [nn.Hardtanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # idx = self.PE(idx)

        # input = torch.cat([input, idx], dim=1)

        x = self.model(input)

        # x_t = x.transpose(2, 3)
        # x = (x + x_t)/2

        return x

class PositionEmbedding(nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(16, 256)
        self.linear = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, True),
            nn.LayerNorm(1024),
            nn.Linear(1024, 4096),
            nn.LayerNorm(4096)
        )

    def forward(self, input):
        input_e = self.embedding(input)
        # if my_util.val and (input == 0 or input == 8):
        #     print(input, input_e)
        input = self.linear(input_e)
        return input.view(-1, 1, 64, 64)

class MyIResNetGenerator(nn.Module):
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
        super(MyIResNetGenerator, self).__init__()

        self.PE = PositionEmbedding()

        model = [IResNet(input_nc + 1, output_nc)]

        model += [nn.Tanh()]
        # model += [nn.Hardtanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input, idx):
        """Standard forward"""
        idx = self.PE(idx)

        input = torch.cat([input, idx], dim=1)

        x = self.model(input)
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

        # model = [UNetD(input_nc, 1, True)]
        
        # self.model = nn.Sequential(*model)

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


class MyPixelDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(MyPixelDiscriminator, self).__init__()

        ndf = 64
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)), #(1, 256, 256) -> (64, 128, 128) # (1, 64, 64) -> (64, 32, 32)
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)), #(64, 128, 128) -> (128, 64, 64) # (64, 32, 32) -> (128, 16, 16)
            norm_layer(ndf*2, affine=True),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)), #(128, 64, 64) -> (256, 32, 32) # (128, 16, 16) -> (256, 8, 8)
            norm_layer(ndf*4, affine=True),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)), #(256, 32, 32) -> (512, 16, 16) # (256, 8, 8) -> (512, 4, 4)
            norm_layer(ndf*8, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, padding=0), # (512, 4, 4) -> (1, 1, 1)
            # nn.Conv2d(ndf*8, 8, kernel_size=4, stride=4, padding=0), # (512, 4, 4) -> (512, 1, 1)
            # nn.LayerNorm(ndf*8),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ndf*8, 8, kernel_size=1)
        )

        """
            nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1), #(512, 16, 16) -> (1024, 8, 8) # (512, 4, 4) -> (1, 1, 1)
            norm_layer(ndf*16),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*16, ndf*32, kernel_size=4, stride=2, padding=1), #(1024, 8, 8) -> (2048, 4, 4)
            norm_layer(ndf*32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*32, 1, kernel_size=3, stride=3, padding=0) #(2048, 4, 4) -> (1, 1, 1)   
        )"""

    def forward(self, x, y):
        # return self.model(x).view(-1, 1)
        out = self.model(x)
        # out = out.view(out.size(0), -1)  # (batch, num_domains)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return out

class LNorm(nn.Module):
    def __init__(self, dim):
        super(LNorm, self).__init__()

    def forward(self, x):
        x = nn.functional.layer_norm(x, x.size()[1:])
        return x


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        affine = True
        instance = True
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=affine, track_running_stats=instance),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=affine, track_running_stats=instance),
            # Shrinkage(dim_out)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = x + self.main(x)
        return x

class ResidualDown(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualDown, self).__init__()
        norm_layer = nn.BatchNorm2d
        '''
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(dim_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out))
        self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=2)
        self.out = nn.LeakyReLU(0.2, True)
        '''
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out))

    def forward(self, x):
        x = x + self.main(x)
        return x
'''
class MyPixelDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(MyPixelDiscriminator, self).__init__()

        ndf = 64
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), # (1, 64, 64) -> (64, 32, 32)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1), # (64, 32, 32) -> (128, 16, 16)
            norm_layer(ndf*2, affine=True),
            nn.LeakyReLU(0.2, True),
            ResidualDown(ndf*2, ndf*2),
            nn.LeakyReLU(0.2, True),
            ResidualDown(ndf*2, ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1), # (128, 16, 16) -> (256, 8, 8)
            norm_layer(ndf*4, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1), # (256, 8, 8) -> (512, 4, 4)
            norm_layer(ndf*8, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=4, padding=0), # (512, 4, 4) -> (512, 1, 1)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)
'''

class ResGroupBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResGroupBlock, self).__init__()
        affine = False
        instance = False
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out*2, kernel_size=1),
            nn.InstanceNorm2d(dim_out, affine=affine, track_running_stats=instance),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out*2, dim_out*2, kernel_size=3, padding=1, groups=dim_out//32),
            nn.InstanceNorm2d(dim_out*2, affine=affine, track_running_stats=instance),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out*2, dim_out, kernel_size=1),
            nn.InstanceNorm2d(dim_out, affine=affine, track_running_stats=instance))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, input_nc, output_nc, conv_dim=64, repeat_num=9):
        super(Generator, self).__init__()

        affine = True
        instance = True
        layers = []
        layers.append(spectral_norm(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=affine, track_running_stats=instance))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(1):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=affine, track_running_stats=instance))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.down = nn.Sequential(*layers)

        layers = []
        for i in range(1):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=affine, track_running_stats=instance))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        
        # Bottleneck layers.
        for i in range(repeat_num):
            '''layers.append(Dense(curr_dim, curr_dim*2))
            layers.append(spectral_norm(nn.Conv2d(curr_dim*2, curr_dim, kernel_size=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim, affine=affine, track_running_stats=instance))
            layers.append(nn.ReLU(inplace=True))'''
            layers.append(ResidualBlock(curr_dim, curr_dim))
            # layers.append(Self_Attn(curr_dim))

        # Up-sampling layers.
        for i in range(1):
            layers.append(spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=affine, track_running_stats=instance))
            # layers.append(nn.Conv2d(in_channels=curr_dim, out_channels=curr_dim//2*4, kernel_size=3, stride=1, padding=1, bias=False))
            # layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        
        self.main = nn.Sequential(*layers)

        layers = []
        for i in range(1):
            layers.append(spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=affine, track_running_stats=instance))
            # layers.append(nn.Conv2d(in_channels=curr_dim, out_channels=curr_dim//2*4, kernel_size=3, stride=1, padding=1, bias=False))
            # layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        self.up = nn.Sequential(*layers)
        
        layers = []
        layers.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.out = nn.Sequential(*layers)

        # self.tanh = nn.Tanh()

        self.pas = nn.MaxPool2d(2)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        # x0 = self._shortcut(x)
        # x = torch.cat([x, x0], dim=1)
        # x0 = self._shortcut(x)
        x2 = self.down(x)
        x1 = self.main(x2)
        x1 = self._pad(x1, x2)
        x1 = self.up(x1)
        x1 = self._pad(x1, x)
        x1 = self.out(x1)

        # x = self.tanh(self._shortcut(x[:, 0]) + x1)

        x_t = x1.transpose(2, 3)
        x = (x1 + x_t)/2

        return x

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
        return x

    def _pad(self, x1, x2):
        assert len(x1[0][0]) <= len(x2[0][0])
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x1

class Ge(nn.Module):
    """Generator network."""
    def __init__(self, input_nc, output_nc, conv_dim=128, repeat_num=8):
        super(Ge, self).__init__()

        affine = False
        instance = False
        layers = []
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=affine, track_running_stats=instance))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.

        layers.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        # x0 = self._shortcut(x)
        # x = torch.cat([x, x0], dim=1)
        x = self.main(x)
        x_t = x.transpose(2, 3)
        x = (x + x_t)/2
        return x

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
        return x

class AdaIN(nn.Module):
    def __init__(self, dim):
        super(AdaIN, self).__init__()

        indim = dim
        outdim = dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mean = nn.Conv2d(indim, outdim, kernel_size=1)
        self.var = nn.Conv2d(indim, outdim, kernel_size=1)
        self.LN = nn.LayerNorm([dim, 1, 1])
        self.relu = nn.ReLU(True)
        self.IN = nn.InstanceNorm2d(outdim)

    def forward(self, content):
        style_pool = self.pool(content)
        mean = self.mean(style_pool)
        mean = self.relu(self.LN(mean))
        var = self.var(style_pool)
        var = self.relu(self.LN(var))
        content = self.IN(content)

        return mean*content + var

class AdaINResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(AdaINResidualBlock, self).__init__()
        affine = False
        instance = False
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            AdaIN(dim_out),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            AdaIN(dim_out),
            # Shrinkage(dim_out)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = x + self.main(x)
        return x

class Shrinkage(nn.Module):
    """Citation:
    Zhao, Minghang, et al. 
    "Deep residual shrinkage networks for fault diagnosis." 
    IEEE Transactions on Industrial Informatics 16.7 (2019): 4681-4690.
    """
    def __init__(self, channel):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.LayerNorm([channel, 1, 1]),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_abs = torch.abs(x)
        x_gap = self.gap(x_abs)
        x_fc = self.fc(x_gap)
        # soft thresholding
        sub = x_abs - x_fc
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.sign(x)*n_sub
        return x

class Dense(nn.Module):
    def __init__(self, indim, outdim, n=4):
        super(Dense, self).__init__()
        assert (indim + indim) == outdim
        self.model = nn.ModuleList([])
        for i in range(n):
            self.model.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(indim + i*indim//n, indim//n, kernel_size=3, padding=1)),
                    nn.InstanceNorm2d(indim//n, affine=True, track_running_stats=True),
                    nn.ReLU(True)
                )
            )
    
    def forward(self, x0):
        x1 = x0
        for conv in self.model:
            x0 = x1
            x1 = conv(x0)
            x1 = torch.cat([x0, x1], dim=1)
        return x1

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out