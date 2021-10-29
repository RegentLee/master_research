from .unet_parts import *
from util import my_util


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # self.inc = DoubleConv(n_channels, 64)
        Block = BasicBlock
        n = 1
        # self.inc = DownBlock(n_channels, 64, Block, n)
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, True),
            # nn.ReflectionPad2d(3),
            # nn.Conv2d(n_channels, 64, kernel_size=7, padding=0),
            # nn.InstanceNorm2d(64),
            # nn.LeakyReLU(0.2, True),
            DownBlock(64, 64, Block, n),
        )
        self.down1 = Down(64, 128, Block, n)
        self.down2 = Down(128, 256, Block, n)
        self.down3 = Down(256, 512, Block, n)
        self.down4 = Down(512, 1024, Block, n)
        """self.down1 = HaloDown(64, 128)
        self.down2 = HaloDown(128, 256)
        self.down3 = HaloDown(256, 512)
        # self.down4 = HaloDown(512, 1024)"""
        factor = 1# 2 if bilinear else 1
        self.down5 = Down(1024, 2048, Block, n)
        # self.down6 = Down(2048, 4096, Block, n)
        self.down6 = Inmost(2048, 4096, Block, n)
        # self.vae = VAE(1024, 2048)
        # self.down6 = Down(2048, 4096, Block, n)
        # sself.down7 = Down(4096, 4096*2, Block, n)
        # self.down8 = Down(4096*2, 4096*4, Block, n)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.u = Up(4096*4, 4096*2 // factor, Block, n, bilinear=bilinear)
        self.p = Up(1024, 1024 // factor, Block, n, bilinear=bilinear)
        self.up = Up(4096, 2048 // factor, Block, n, bilinear=bilinear)
        self.up0 = Up(2048, 1024 // factor, Block, n, bilinear=bilinear)  
        self.up1 = Up(1024, 512 // factor, Block, n, bilinear=bilinear)
        self.up2 = Up(512, 256 // factor, Block, n, bilinear=bilinear)
        self.up3 = Up(256, 128 // factor, Block, n, bilinear=bilinear)
        self.up4 = Up(128, 64, Block, n, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

        self.attn = Attention(1)

    def forward(self, x):
        x0 = self._shortcut(x[:, 0])
        x1 = self.inc(x) # (1, 64, 64) -> (64, 64, 64)
        x2 = self.down1(x1) # (64, 64, 64) -> (128, 32, 32)
        x3 = self.down2(x2) # (128, 32, 32) -> (256, 16, 16)
        x4 = self.down3(x3) # (256, 16, 16) -> (512, 8, 8)
        # x5 = self.down4(x4) # (512, 8, 8) -> (1024, 4, 4)
        # x6 = self.down5(x5) # (1024, 4, 4) -> (2048, 2, 2)
        # x6 = self.down6(x6) # (2048, 2, 2) -> (4096, 1, 1) -> (2048, 2, 2)
        # x = self.up(x, x6)
        # x = self.up0(x6, x5)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = logits + x0
        return logits

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
        return x
'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = Encoder(n_channels)
        self.decoder = Decoder(n_classes)
        

    def forward(self, x):
        logits = self.decoder(self.encoder(x))
        return logits

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
        return x
'''
class Encoder(nn.Module):
    def __init__(self, n_channels):
        super(Encoder, self).__init__()
        
        ngf = 64
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, ngf, kernel_size=1, bias=False), # (1, 64, 64) -> (64, 64, 64)
            # nn.InstanceNorm2d(ngf),
            # nn.LeakyReLU(0.2, True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1), # (64, 64, 64) -> (128, 32, 32)
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1), # (128, 32, 32) -> (256, 16, 16)
            nn.InstanceNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1), # (256, 16, 16) -> (512, 8, 8)
            nn.InstanceNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*16, kernel_size=4, stride=2, padding=1), # (512, 8, 8) -> (1024, 4, 4)
            nn.InstanceNorm2d(ngf*16),
            nn.LeakyReLU(0.2, True),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(ngf*16, ngf*32, kernel_size=4, stride=2, padding=1), # (1024, 4, 4) -> (2048, 2, 2)
            nn.InstanceNorm2d(ngf*32),
            nn.LeakyReLU(0.2, True),
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(ngf*32, ngf*64, kernel_size=4, stride=2, padding=1), # (2048, 2, 2) -> (4096, 1, 1)
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x0 = self.inc(x) # (1, 64, 64) -> (64, 64, 64)
        x1 = self.down1(x0) # (64, 64, 64) -> (128, 32, 32)
        x2 = self.down2(x1) # (128, 32, 32) -> (256, 16, 16)
        x3 = self.down3(x2) # (256, 16, 16) -> (512, 8, 8)
        x4 = self.down4(x3) # (512, 8, 8) -> (1024, 4, 4)
        x5 = self.down5(x4) # (1024, 4, 4) -> (2048, 2, 2)
        x6 = self.down6(x5) # (2048, 2, 2) -> (4096, 1, 1)
        return x1, x2, x3, x4, x5, x6

class Decoder(nn.Module):
    def __init__(self, n_classes):
        super(Decoder, self).__init__()
        
        ngf = 64
        self.outc = nn.Sequential(
            nn.Conv2d(ngf, n_classes, kernel_size=1, bias=False), # (1, 64, 64) <- (64, 64, 64)
            # nn.InstanceNorm2d(ngf),
            # nn.LeakyReLU(0.2, True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1), # (64, 64, 64) <- (128, 32, 32)
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4*2, ngf*2, kernel_size=4, stride=2, padding=1), # (128, 32, 32) <- (256, 16, 16)
            nn.InstanceNorm2d(ngf*2, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1), # (256, 16, 16) <- (512, 8, 8)
            nn.InstanceNorm2d(ngf*4, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16*2, ngf*8, kernel_size=4, stride=2, padding=1), # (512, 8, 8) <- (1024, 4, 4)
            nn.InstanceNorm2d(ngf*8, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*32*2, ngf*16, kernel_size=4, stride=2, padding=1), # (1024, 4, 4) <- (2048, 2, 2)
            nn.InstanceNorm2d(ngf*16, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*64, ngf*32, kernel_size=4, stride=2, padding=1), # (2048, 2, 2) <- (4096, 1, 1)
            nn.InstanceNorm2d(ngf*32, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, code):
        x1, x2, x3, x4, x5, x6 = code
        x = self.up6(x6) # (2048, 2, 2) <- (4096, 1, 1)
        x = self.up5(torch.cat([x5, x], dim=1)) # (1024, 4, 4) <- (2048, 2, 2)
        x = self.up4(torch.cat([x4, x], dim=1)) # (512, 8, 8) <- (1024, 4, 4)
        x = self.up3(torch.cat([x3, x], dim=1)) # (256, 16, 16) <- (512, 8, 8)
        x = self.up2(torch.cat([x2, x], dim=1)) # (128, 32, 32) <- (256, 16, 16)
        x = self.up1(torch.cat([x1, x], dim=1)) # (64, 64, 64) <- (128, 32, 32)
        x = self.outc(x) # (1, 64, 64) <- (64, 64, 64)
        return x


class UNetD(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        Block = ResNeXt
        self.inc = DownBlock(n_channels, 64, Block, 1)
        self.down1 = Down(64, 128, Block, 1)
        self.down2 = Down(128, 256, Block, 1)
        self.down3 = Down(256, 512, Block, 1)
        self.down4 = Down(512, 1024, Block, 1)
        factor = 1# 2 if bilinear else 1
        self.down4 = Down(512, 1024, Block, 1)
        self.down5 = Down(1024, 2048, Block, 1)
        self.up0 = Up(2048, 1024 // factor, Block, 1, bilinear=bilinear)  
        self.up1 = Up(1024, 512 // factor, Block, 1, bilinear=bilinear)
        self.up2 = Up(512, 256 // factor, Block, 1, bilinear=bilinear)
        self.up3 = Up(256, 128 // factor, Block, 1, bilinear=bilinear)
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            DownBlock(64, 64, Block, 1)
        )
        Up(128, 64, Block, 1, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

        self.out_all = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 1, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x)
        logits = self.outc(x)
        logits_all = self.out_all(x6)
        return logits, logits_all


'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            DenseDown(32, 64)
        )
        self.down1 = Down(64, 128, DenseDown, 1)
        self.down2 = Down(128, 256, DenseDown, 1)
        self.down3 = Down(256, 512, DenseDown, 1)
        self.down4 = Down(512, 1024, DenseDown, 1)
        factor = 1# 2 if bilinear else 1
        self.down4 = Down(512, 1024, DenseDown, 1)
        self.up1 = DenseUp(1024, 512)

        self.up2 = DenseUp(512, 256)

        self.up3 = DenseUp(256, 128)

        self.up4 = DenseUp(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x0 = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x5 = self.down5(x5)
        # x = self.up0(x6, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) + x0
        return logits
'''


class IResNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(IResNet, self).__init__()

        ngf = 64
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(n_channels, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.LeakyReLU(0.2, True)]

        model += [ResGroup(ngf, ngf), ResGroup(ngf, ngf), ResGroup(ngf, ngf)]
        model += [ResGroupDown(ngf, ngf*2)]
        model += [ResGroup(ngf*2, ngf*2), ResGroup(ngf*2, ngf*2), ResGroup(ngf*2, ngf*2)]
        model += [ResGroupDown(ngf*2, ngf*4)]
        model += [ResGroup(ngf*4, ngf*4), ResGroup(ngf*4, ngf*4), ResGroup(ngf*4, ngf*4)]

        model += [
            nn.Conv2d(ngf*4, ngf*8, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        ]

        model += [ResGroup(ngf, ngf), ResGroup(ngf, ngf)]

        model += [OutConv(ngf, n_classes)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) + self._shortcut(x[:, 0])

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
        return x




        