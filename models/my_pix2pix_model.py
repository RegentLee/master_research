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

from torchvision.models.resnet import BasicBlock

from util import my_util
from .unet import UNet

from torchinfo import summary


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
        self.loss_names = ['G_GAN', 'G_CE', 'D']# 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'image', 'real_B_0']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
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
        summary(self.net, input_size=(1, 1, 64, 64), depth=10)
        # print(self.net.state_dict()["linear.weight"])
        t = torch.tensor([[1e-9] + my_util.distance], dtype=torch.float32)
        self.net.state_dict()["linear.weight"][:] = t# .to(self.device)
        # for param_tensor in net.state_dict():
        #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # netD = ResNet(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, norm_layer=networks.get_norm_layer(norm_type=opt.norm))
            # self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
            summary(self.netD, input_size=(1, 2, 64, 64))

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCE = my_util.CustomCELoss() # torch.nn.CrossEntropyLoss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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

        self.real_B_0 = torch.div(self.real_B, my_util.distance[-1])*2 - 1
        self.real_B_0 = self.real_B_0.to(torch.float)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.fake, self.image, self.prob = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.image), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        self.real_B_0 = torch.div(self.real_B, my_util.distance[-1])*2 - 1
        self.real_B_0 = self.real_B_0.to(torch.float)
        real_AB = torch.cat((self.real_A, self.real_B_0), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # print(self.fake_B[0].shape)
        # print(torch.flatten(self.real_B).shape)
        # print(self.real_B)
        real_B_CE = torch.div(self.real_B - 2, 0.32).to(torch.long) + 1
        real_B_CE = torch.where(real_B_CE < 0, 0, real_B_CE)
        real_B_CE = torch.where(real_B_CE > 63, 63, real_B_CE)
        real_B_CE = real_B_CE.to(torch.long)
        # print(real_B_CE)
        self.loss_G_CE = self.criterionCE(self.fake[0], torch.flatten(real_B_CE))
        self.loss_G_L1 = self.loss_G_CE * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

        # print("Model's state_dict:")
        # print(self.net.state_dict()["linear.weight"])
        # print(self.net.state_dict()["linear.bias"])

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
            BasicBlock(64, 64, norm_layer=norm_layer),

            BasicBlock(64, 128, 2, downsample=nn.Conv2d(64, 128, 1, stride=2), norm_layer=norm_layer),
            BasicBlock(128, 128, norm_layer=norm_layer),
            BasicBlock(128, 128, norm_layer=norm_layer),
            BasicBlock(128, 128, norm_layer=norm_layer),

            BasicBlock(128, 256, 2, downsample=nn.Conv2d(128, 256, 1, stride=2), norm_layer=norm_layer),
            BasicBlock(256, 256, norm_layer=norm_layer),
            BasicBlock(256, 256, norm_layer=norm_layer),
            BasicBlock(256, 256, norm_layer=norm_layer),
            BasicBlock(256, 256, norm_layer=norm_layer),
            BasicBlock(256, 256, norm_layer=norm_layer),

            BasicBlock(256, 512, 2, downsample=nn.Conv2d(256, 512, 1, stride=2), norm_layer=norm_layer),
            BasicBlock(512, 512, norm_layer=norm_layer),
            BasicBlock(512, 512, norm_layer=norm_layer),

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
