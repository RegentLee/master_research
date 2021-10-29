""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            # BasicBlock(in_channels, out_channels),
            # BasicBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, True),
            # SEBlock(out_channels),
        )
        self.attn = CrissCrossAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.attn(self.attn(x))
        return x

class ConvDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout()
            # SEBlock(out_channels),
        )
        self.attn = CrissCrossAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.attn(self.attn(x))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, num=2, affine=False, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            # print(in_channels, mid_channels)
        model = [block(in_channels, mid_channels, affine)]
        for i in range(num - 1):
            model += [block(mid_channels, out_channels, affine)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block, num=2):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            # nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            # LNorm(out_channels),
            nn.LeakyReLU(0.2, True),
            # ResGroupDown(in_channels, out_channels),
            DownBlock(out_channels, out_channels, block, num)
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            # nn.InstanceNorm2d(in_channels),
            # nn.LeakyReLU(0.2, True),
            # DownBlock(in_channels, out_channels, block, num)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block, num=2, bilinear=True):
        super().__init__()

        # self.a = nn.parameter.Parameter(torch.Tensor(0.5))

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
            )
        if in_channels == out_channels:
            in_channels = in_channels*2
        self.conv1 = nn.Sequential(
            DownBlock(in_channels, out_channels, block, num, affine=True)
        )
        self.conv2 = nn.Sequential(
            DownBlock(out_channels, out_channels, block, num, affine=True)
        )
        '''if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
            self.conv = DownBlock(in_channels, out_channels, block, num, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DownBlock(in_channels, out_channels, block, num)'''
        '''self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.up2 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DownBlock(in_channels, out_channels, block, num)'''


    def forward(self, x1, x2=None):
        x1 = self.up1(x1)
        if torch.is_tensor(x2):
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv1(x)
        else:
            return self.conv2(x1)


class Inmost(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block, num=2, bilinear=True):
        super().__init__()

        # self.a = nn.parameter.Parameter(torch.Tensor(0.5))

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.up = nn.Sequential(
                nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)
            )
        if in_channels == out_channels:
            in_channels = in_channels*2
        self.conv = nn.Sequential(
            DownBlock(out_channels, in_channels, block, num)
        )     

    def forward(self, x1):
        x2 = x1
        x1 = self.down(x1)
        # x1 = self.dropout(x1)
        x1 = self.up(x1)
        # x2 = self.attn(x1, x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAE, self).__init__()

        input_dim = out_channels*4
        noise_dim = out_channels

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

        self.input = nn.Flatten()
        self.fc_mu = nn.Linear(input_dim, noise_dim)
        self.fc_var = nn.Linear(input_dim, noise_dim)
        self.decoder_input = nn.Linear(noise_dim, input_dim)

    def forward(self, x):
        x = self.down(x)
        x0 = x
        x = self.input(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder_input(z)
        self.mu = mu
        self.log_var = log_var
        return  output.view(x0.shape)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        mu = self.mu
        log_var = self.log_var

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        return kld_loss

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x


class ResNeXt(nn.Module):
    def __init__(self, indim, outdim):
        super(ResNeXt, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Sequential(
                nn.Conv2d(indim, outdim, kernel_size=1),
                nn.InstanceNorm2d(outdim)
            )
            
        
        if indim > outdim:
            dim_inter = indim // 2
        else:
            dim_inter = outdim // 2
        model = [nn.Conv2d(indim, dim_inter, 1), 
                 nn.InstanceNorm2d(dim_inter),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(dim_inter, dim_inter, 3, padding=1, groups=32),
                 nn.InstanceNorm2d(dim_inter),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(dim_inter, outdim, 1),
                 nn.InstanceNorm2d(outdim),
                 # Shrinkage(outdim),
        ]

        self.model = nn.Sequential(*model)

        self.SE = SEBlock(outdim)
        # self.SE = Attention(outdim)

        self.relu = nn.LeakyReLU(0.2, True)
        

    def forward(self, x):
        shortcut = x

        out = self.model(x)
        # out = self.SE(out)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, indim, outdim, affine=False):
        super(BasicBlock, self).__init__()
        track_running_stats = affine
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Sequential(
                nn.Conv2d(indim, outdim, kernel_size=1),
                nn.InstanceNorm2d(outdim, affine=affine, track_running_stats=track_running_stats)
            )
        
        model = [nn.Conv2d(indim, outdim, kernel_size=3, padding=1), 
                 nn.InstanceNorm2d(outdim, affine=affine, track_running_stats=track_running_stats),
                 # LNorm(outdim),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(outdim, outdim, kernel_size=3, padding=1),
                 nn.InstanceNorm2d(outdim, affine=affine, track_running_stats=track_running_stats),
                 # LNorm(outdim),
                 # SEBlock(outdim),
                 # Shrinkage(outdim),
        ]

        self.model = nn.Sequential(*model)

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        shortcut = x

        out = self.model(x)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResGroup(nn.Module):
    def __init__(self, indim, outdim, affine=False):
        super(ResGroup, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Sequential(
                nn.Conv2d(indim, outdim, kernel_size=1),
                nn.InstanceNorm2d(outdim, affine=affine)
            )
        
        dim_inter = outdim*2
        model = [nn.Conv2d(indim, dim_inter, 1), 
                 nn.InstanceNorm2d(dim_inter, affine=affine),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(dim_inter, dim_inter, 3, padding=1, groups=dim_inter//32),
                 nn.InstanceNorm2d(dim_inter, affine=affine),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(dim_inter, outdim, 1),
                 nn.InstanceNorm2d(outdim, affine=affine),
        ]

        self.model = nn.Sequential(*model)

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        shortcut = x

        out = self.model(x)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResGroupDown(nn.Module):
    def __init__(self, indim, outdim):
        super(ResGroupDown, self).__init__()
        self.shortcut = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(indim, outdim, kernel_size=1),
            nn.InstanceNorm2d(outdim)
        )
        
        dim_inter = outdim*2
        model = [nn.Conv2d(indim, dim_inter, 1), 
                 nn.InstanceNorm2d(dim_inter),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(dim_inter, dim_inter, 3, padding=1, groups=dim_inter//32, stride=2),
                 nn.InstanceNorm2d(dim_inter),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(dim_inter, outdim, 1),
                 nn.InstanceNorm2d(outdim),
        ]

        self.model = nn.Sequential(*model)

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        shortcut = x

        out = self.model(x)
        
        # Projection shortcutの場合
        shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

class NoBlock(nn.Module):
    def __init__(self, indim, outdim):
        super(NoBlock, self).__init__()
        self.dim = indim != outdim
        if self.dim:
            self.model = nn.Conv2d(indim, outdim, kernel_size=1)

    def forward(self, x):
        if self.dim:
            x = self.model(x)
        return x

'''
class Attention(nn.Module):
    """Citation:
    Zhang, Han, et al. 
    "Self-attention generative adversarial networks." 
    *International conference on machine learning*. PMLR, 2019.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.theta = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=1),
            # nn.BatchNorm2d(dim//8)
        )

        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=1),
            # nn.BatchNorm2d(dim//8)
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.softmax = nn.Softmax(dim=1)

        self.g = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1),
            # nn.BatchNorm2d(dim//2)
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.v = nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=1),
            # nn.BatchNorm2d(dim)
        )


    def forward(self, q, kv=None):
        if not torch.is_tensor(kv):
            kv = q
        assert q.shape == kv.shape
        batch_size, C, H, W = kv.shape
        location_num = H*W
        downsampled_num = location_num //4

        theta = self.theta(q)
        theta = theta.view([batch_size, C//8, location_num])

        phi = self.phi(kv)
        phi = phi.view([batch_size, C//8, downsampled_num])

        attn = torch.matmul(phi.transpose(1, 2), theta)
        attn = self.softmax(attn)

        g = self.g(kv)
        g = g.view([batch_size, C//2, downsampled_num])

        attn_g = torch.matmul(g, attn)
        attn_g = attn_g.view([batch_size, C//2, H, W])

        attn_g = self.v(attn_g)

        return attn_g
'''

class Attention(nn.Module):
    """Citation:
    Oktay, Ozan, et al. 
    "Attention u-net: Learning where to look for the pancreas." 
    arXiv preprint arXiv:1804.03999 (2018).
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(dim, dim*16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim*16)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(dim, dim*16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim*16)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(dim*16, 1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(0.2, True)


    def forward(self, g, x=None):
        if not torch.is_tensor(x):
            x = g
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi

'''
class SEBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.g = nn.Conv2d(dim, dim//16, kernel_size=1)  # C*H*W -> (C/16)*H*W

        self.x = nn.Conv2d(dim, dim//16, kernel_size=1)  # C*H*W -> (C/16)*H*W

        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(0.2, True)

        self.psi = nn.Sequential(
            nn.Conv2d(dim//16, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x=None):
        if not torch.is_tensor(x):
            x = g
        x0 = x
        g = self.g(g)
        x = self.x(x)
        psi = self.relu((g + x)/2)
        psi = self.psi(psi)
        return psi*x0
'''

class SEBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C*H*W -> C*1*1
            nn.Conv2d(dim, dim//16, kernel_size=1),  # C*1*1 -> (C/16)*1*1
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//16, dim, kernel_size=1),  # (C/16)*1*1 -> C*1*1
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x0 = x
        x = self.model(x)
        return x0*x


class Encoder(nn.Module):
    def __init__(self, indim, outdim):
        super(Encoder, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Conv2d(indim, outdim, 1)

        self.Attention = Attention(indim)
        self.AttentionAfter = nn.Sequential(
            nn.InstanceNorm2d(indim),
            nn.ReLU(inplace=True),
        )

        dim_inter = int(outdim / 2)
        self.FFN = nn.Sequential(
            nn.Conv2d(indim, dim_inter, 1), 
            nn.InstanceNorm2d(dim_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_inter, dim_inter, 3, padding=1, groups=32),
            nn.InstanceNorm2d(dim_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_inter, outdim, 1),
            nn.InstanceNorm2d(outdim),
        )
        self.FFNAfter = nn.Sequential(
            nn.ReLU(inplace=True),
        )    

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.AttentionAfter(attn + x)

        out = self.FFN(attn)
        # Projection shortcutの場合
        if self.is_dim_changed:
            attn = self.shortcut(attn)
        out = self.FFNAfter(out + attn)

        return out


class Decoder(nn.Module):
    def __init__(self, indim, outdim):
        super(Decoder, self).__init__()
        self.is_dim_changed = (indim//2 != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Conv2d(indim, outdim, 1)

        self.up = nn.ConvTranspose2d(indim, indim // 2, kernel_size=2, stride=2)

        dim_inter = int(outdim / 2)
        self.SelfAttention = nn.Sequential(
            nn.Conv2d(indim//2, dim_inter, 1), 
            nn.InstanceNorm2d(dim_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_inter, dim_inter, 3, padding=1, groups=32),
            nn.InstanceNorm2d(dim_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_inter, outdim, 1),
            nn.InstanceNorm2d(indim//2),
            SEBlock(outdim)
        )
        # Attention(indim//2)
        self.SAAfter = nn.Sequential(
            nn.ReLU(inplace=True),
        )

        self.CrossAttention = Attention(indim//2)
        self.CAAfter  = nn.Sequential(
            nn.BatchNorm2d(indim//2),
            nn.ReLU(inplace=True),
        )

        dim_inter = int(outdim / 2)
        self.FFN = nn.Sequential(
            nn.Conv2d(indim, dim_inter, 1), 
            nn.InstanceNorm2d(dim_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_inter, dim_inter, 3, padding=1, groups=32),
            nn.InstanceNorm2d(dim_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_inter, outdim, 1),
            nn.InstanceNorm2d(outdim),
            SEBlock(outdim)
        )
        self.FFNAfter = nn.Sequential(
            nn.ReLU(inplace=True),
        )    

    def forward(self, x1, x2):
        x1 = self.up(x1)
        attn = self.SelfAttention(x1)
        attn = self.SAAfter(attn + x1)

        cattn = self.CrossAttention(attn, x2)
        cattn = self.CAAfter(cattn) # + attn)

        x = torch.cat([cattn, attn], dim=1)

        out = self.FFN(x)
        # Projection shortcutの場合
        # if self.is_dim_changed:
        #     cattn = self.shortcut(cattn)
        out = self.FFNAfter(out)#out + cattn)

        return out


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class DenseDown(nn.Module):
    def __init__(self, indim, outdim, n=2):
        super(DenseDown, self).__init__()
        assert (indim + indim) == outdim
        self.model = nn.ModuleList([])
        for i in range(n):
            self.model.append(
                nn.Sequential(
                    nn.Conv2d(indim + i*indim//n, indim//n, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(indim),
                    nn.LeakyReLU(0.2, True)
                )
            )
    
    def forward(self, x0):
        x1 = x0
        for conv in self.model:
            x0 = x1
            x1 = conv(x0)
            x1 = torch.cat([x0, x1], dim=1)
        return x1

class DenseUp(nn.Module):
    def __init__(self, indim, outdim):
        super(DenseUp, self).__init__()
        self.up = nn.Sequential(
                nn.Conv2d(indim, outdim*4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )

        self.bottleneck1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv = DenseDown(outdim, indim)

        self.bottleneck2 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x0 = torch.cat([x2, x1], dim=1)
        x0 = self.bottleneck1(x0)
        x0 = self.conv(x0)
        x0 = self.bottleneck2(x0)
        return x0


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


class LNorm(nn.Module):
    def __init__(self, dim):
        super(LNorm, self).__init__()

    def forward(self, x):
        x = nn.functional.layer_norm(x, x.size()[1:])
        return x


class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.Identity()

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        #
        out = self.relu(x_master + x_gpb)

        return out


class FPAup(nn.Module):
    def __init__(self, in_channels, out_channels, block, num):
        super(FPAup, self).__init__()

        self.conv = DownBlock(in_channels*2, out_channels, block, num)

    def forward(self, x1, x2=None):
        if torch.is_tensor(x2):
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        else:
            return self.conv2(x1)