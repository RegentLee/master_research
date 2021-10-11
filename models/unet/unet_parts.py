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
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            BasicBlock(in_channels, out_channels),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            # SEBlock(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, num=2):
        super().__init__()
        model = [block(in_channels, out_channels)]
        for i in range(num - 1):
            model += [block(out_channels, out_channels)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block, num=2):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            DownBlock(in_channels, out_channels, block, num)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block, num=2, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DownBlock(in_channels, out_channels, block, num)


    def forward(self, x1, x2):
        x1 = self.up(x1)
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNeXt(nn.Module):
    def __init__(self, indim, outdim):
        super(ResNeXt, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Conv2d(indim, outdim, 1)
        
        dim_inter = int(outdim / 4)
        model = [nn.Conv2d(indim, dim_inter, 1), 
                 nn.InstanceNorm2d(dim_inter),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(dim_inter, dim_inter, 3, padding=1),# , groups=32),
                 nn.InstanceNorm2d(dim_inter),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(dim_inter, outdim, 1),
                 nn.InstanceNorm2d(outdim)
        ]

        self.model = nn.Sequential(*model)

        self.SE = SEBlock(outdim)
        # self.SE = Attention(outdim)

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        shortcut = x

        out = self.model(x)
        out = self.SE(out)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, indim, outdim):
        super(BasicBlock, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Conv2d(indim, outdim, kernel_size=1)
        
        model = [nn.Conv2d(indim, outdim, kernel_size=3, padding=1), 
                 nn.InstanceNorm2d(outdim),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(outdim, outdim, kernel_size=3, padding=1),
                 nn.InstanceNorm2d(outdim),
        ]

        self.model = nn.Sequential(*model)

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        shortcut = x

        out = self.model(x)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

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
            nn.Conv2d(dim, dim//2, kernel_size=1),
            nn.InstanceNorm2d(dim//2)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1),
            nn.InstanceNorm2d(dim//2)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(dim//2, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x=None):
        if not torch.is_tensor(x):
            x = g
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi


class SEBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.g = nn.Conv2d(dim, dim//16, kernel_size=1)  # C*H*W -> (C/16)*H*W

        self.x = nn.Conv2d(dim, dim//16, kernel_size=1)  # C*H*W -> (C/16)*H*W

        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.PReLU()

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
            # nn.AdaptiveAvgPool2d(1), # C*H*W -> C*1*1
            nn.Conv2d(dim, dim//16, kernel_size=1),  # C*H*W -> (C/16)*H*W
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//16, dim, kernel_size=1),  # (C/16)*H*W -> C*H*W
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x0 = x
        x = self.model(x)
        return x0*x
'''

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