""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_dropblock=False, use_CBAM=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        ###################################################################
        self.tanh = nn.Tanh()     # 无监督分支用
        self.outc = (OutConv(64, n_classes))

        self.outc2 = (OutConv(64, n_classes))
        ###################################################################

        ###################################################################
        # 增加dropblock操作
        self.use_dropblock = use_dropblock
        if use_dropblock:
            self.drop_block1 = DropBlock2D(block_size=13, drop_prob=0.2)
            self.drop_block2 = DropBlock2D(block_size=7, drop_prob=0.2)

        self.use_CBAM = use_CBAM
        if use_CBAM:
            self.cbam = CBAM(64)    # 增加CBAM操作  https://blog.csdn.net/qq_45981086/article/details/130914361
        ###################################################################

    def forward(self, x):
        x1 = self.inc(x)                                                # torch.Size([2, 64, 224, 224])
        if self.use_dropblock:
            x1 = self.drop_block1(x1)  # 增加dropblock操作
        x2 = self.down1(x1)                                             # torch.Size([2, 128, 112, 112])
        if self.use_dropblock:
            x2 = self.drop_block2(x2)  # 增加dropblock操作
        x3 = self.down2(x2)                                             # torch.Size([2, 256, 56, 56])
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x9 = self.up4(x, x1)

        if self.use_CBAM:
            x9 = self.cbam(x9)

        out = self.outc(x9)
        out_tanh = self.tanh(out)

        out_seg = self.outc2(x9)
        return out_tanh, out_seg

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)