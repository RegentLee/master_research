from .unet_parts import *
from util import my_util


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        """self.inc = nn.Sequential(
            DownBlock(n_channels, 64, ConvBlock, 1),
            DownBlock(64, 128, ConvBlock, 1)
        )
        self.down1 = Down(128, 256, Encoder, 1)
        self.down2 = Down(256, 512, Encoder, 1)
        factor = 2 if bilinear else 1
        self.down3 = Down(512, 1024 // factor, Encoder, 1)
        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)

        self.outc = OutConv(128, n_classes)"""
        
        # self.inc = DoubleConv(n_channels, 64)
        Block = BasicBlock
        n = 1
        # self.inc = DownBlock(n_channels, 64, Block, n)
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, True),
            DownBlock(64, 64, Block, n),
        )
        self.down1 = Down(64, 128, Block, n)
        self.down2 = Down(128, 256, Block, n)
        self.down3 = Down(256, 512, Block, n)
        self.down4 = Down(512, 1024, Block, n)
        factor = 1# 2 if bilinear else 1
        self.down4 = Down(512, 1024, Block, n)
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

        """self.inc = nn.Sequential(
            DownBlock(n_channels, 64, ResNeXt, 1),
            DownBlock(64, 128, ResNeXt, 1),
            DownBlock(128, 256, ResNeXt, 2),
        )
        self.down1 = Down(256, 512, ResNeXt, 4)
        self.down2 = Down(512, 1024, ResNeXt, 6)
        factor = 2 if bilinear else 1
        self.down3 = Down(1024, 2048 // factor, ResNeXt, 6)
        self.attn1 = Attention(1024)
        # self.up1 = Up(512, 256 // factor, ConvBlock, 2, bilinear=bilinear)
        self.up1_1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up1_2 = DownBlock(2048, 1024, ResNeXt, 6)

        self.attn2 = Attention(512)
        # self.up2 = Up(256, 128 // factor, ConvBlock, 2, bilinear=bilinear)
        self.up2_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2_2 = DownBlock(1024, 512, ResNeXt, 4)

        self.attn3 = Attention(256)
        # self.up3 = Up(128, 64, ConvBlock, 2, bilinear=bilinear)
        self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3_2 = DownBlock(512, 256, ResNeXt, 3)

        self.outc = OutConv(256, n_classes)"""

    def forward(self, x):
        x0 = self._shortcut(x)
        x1 = self.inc(x) # (1, 64, 64) -> (64, 64, 64)
        x2 = self.down1(x1) # (64, 64, 64) -> (128, 32, 32)
        x3 = self.down2(x2) # (128, 32, 32) -> (256, 16, 16)
        x4 = self.down3(x3) # (256, 16, 16) -> (512, 8, 8)
        x5 = self.down4(x4) # (512, 8, 8) -> (1024, 4, 4)
        # x6 = self.vae(x5)
        # x6 = self.down5(x5) # (1024, 4, 4) -> (2048, 2, 2)
        # x6 = self.down6(x6) # (2048, 2, 2) -> (4096, 1, 1) -> (2048, 2, 2)
        # x8 = self.vae(x7)
        # x7 = self.down6(x6)
        # x8 = self.down7(x7)
        # x9 = self.down8(x8)
        # x6 = self.dropout1(x6)
        # x = self.u(x9, x8)
        # x = self.p(x8, x7)
        # x = self.dropout2(x)
        # x = self.up(x, x6)
        # x = self.up0(x6, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        '''x = self.up1_1(x4)
        # x3 = self.attn1(x, x3)
        x = torch.cat([x3, x], dim=1)
        x = self.up1_2(x)
        x = self.up2_1(x)
        # x2 = self.attn2(x, x2)
        x = torch.cat([x2, x], dim=1)
        x = self.up2_2(x)
        x = self.up3_1(x)
        x1 = self.attn3(x, x1)
        x = torch.cat([x1, x], dim=1)
        x = self.up3_2(x)'''
        logits = self.outc(x)
        # logits_t = logits.transpose(2, 3)
        # logits = (logits + logits_t)/2
        # logits = self.attn(x0, logits)
        # output = logits + x0
        return logits

    def _shortcut(self, x):
        x = (x + 1)*my_util.distance[-1]
        m_max = torch.tensor(my_util.distance[-1], dtype=torch.float, device=x.device)
        m_min = torch.tensor(0, dtype=torch.float, device=x.device)
        x = torch.where(x > m_max, m_max, x)
        x = (x - m_min)/(m_max - m_min)*2 - 1
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