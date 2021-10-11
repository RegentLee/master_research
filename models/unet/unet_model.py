from .unet_parts import *


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
        self.inc = DownBlock(n_channels, 64, ConvBlock, 2)
        self.down1 = Down(64, 128, ResNeXt, 2)
        self.down2 = Down(128, 256, ResNeXt, 2)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor, ConvBlock, 2)
        self.attn1 = SEBlock(256)# Attention(256)
        # self.up1 = Up(512, 256 // factor, ConvBlock, 2, bilinear=bilinear)
        self.up1_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1_2 = DownBlock(512, 256, ConvBlock, 2)

        self.attn2 = SEBlock(128)# Attention(128)
        # self.up2 = Up(256, 128 // factor, ConvBlock, 2, bilinear=bilinear)
        self.up2_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2_2 = DownBlock(256, 128, ConvBlock, 2)

        self.attn3 = SEBlock(64)# Attention(64)
        # self.up3 = Up(128, 64, ConvBlock, 2, bilinear=bilinear)
        self.up3_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3_2 = DownBlock(128, 64, ConvBlock, 2)

        self.attn4 = Attention(64)

        self.outc = OutConv(64, n_classes)

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
        x1 = self.inc(x)
        # x1 = self.attn4(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        '''x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)'''
        x = self.up1_1(x4)
        # x3 = self.attn1(x, x3)
        x = torch.cat([x3, x], dim=1)
        x = self.up1_2(x)
        x = self.up2_1(x)
        # x2 = self.attn2(x, x2)
        x = torch.cat([x2, x], dim=1)
        x = self.up2_2(x)
        x = self.up3_1(x)
        # x1 = self.attn3(x, x1)
        x = torch.cat([x1, x], dim=1)
        x = self.up3_2(x)
        logits = self.outc(x)
        return logits