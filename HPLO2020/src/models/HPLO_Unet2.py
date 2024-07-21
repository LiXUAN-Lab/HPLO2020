"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM
from .PositionalEncoding import FixedPositionalEncoding
from . import Transformer

def myreshape(x, dims):

    x = x.view(
        x.size(0),
        dims[0],
        dims[1],
        dims[2],
        x.size(2),
    )

    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x

class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        # self.gn = normalization(num_in, norm=norm)
        self.gn = norm(num_out)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.conv(x)
        h = self.act_fn(self.gn(h))
        return h

class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        # self.gn = normalization(num_in, norm=norm)
        self.gn = norm(num_out)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.conv(x)
        h = self.act_fn(self.gn(h))
        return h

class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFunit, self).__init__()
        self.norm = norm
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid,num_out,kernel_size=(3,3,3),stride=stride,g=g,d=(d[0],d[0],d[0]),norm=norm) # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(3,3,1),stride=1,g=g,d=(d[1],d[1],1),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):
        # print('MFunit######x',x.shape)
        # print('MFunit######norm',self.norm)
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x

        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut

class DMFUnit(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(DMFUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

        num_mid = num_in if num_in <= num_out else num_out
        self.num_in = num_in
        self.num_mid = num_mid
        self.num_out = num_out
        # multiplexer
        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2,3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_mid,num_out, kernel_size=(3, 3, 3), stride=stride, g=g, d=(dilation[i],dilation[i], dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)


    def forward(self, x):
        # print('DMFUnit#############num_in,num_mid,num_out:', self.num_in, self.num_mid, self.num_out)
        # print('DMFUnit#############:',x.shape)

        x1 = self.conv1x1x1_in1(x)
        # print('    DMFUnit#############x1:', x1.shape)
        x2 = self.conv1x1x1_in2(x1)
        # print('    DMFUnit#############x2:', x2.shape)
        # test1 = self.conv3x3x3_m1[0](x2)
        # test2 = self.conv3x3x3_m1[1](x2)
        # test3 = self.conv3x3x3_m1[2](x2)
        # print('       DMFUnit#############test1:', test1.shape)
        # print('       DMFUnit#############test2:', test2.shape)
        # print('       DMFUnit#############test3:', test3.shape)
        x3 = self.weight1*self.conv3x3x3_m1[0](x2) + self.weight2*self.conv3x3x3_m1[1](x2) + self.weight3*self.conv3x3x3_m1[2](x2)
        # print('    DMFUnit#############x3:', x3.shape)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        return x4 + shortcut

class FTrans(nn.Module):
    def __init__(self, c_in = 256):
        super(FTrans, self).__init__()

        # self.conv3d_add = nn.Conv3d(
        #     c_in,
        #     2 * c_in,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )
        self.position_encoding = FixedPositionalEncoding(c_in)
        self.pe_dropout = nn.Dropout(p=0.1)
        dpr1 = [x.item() for x in torch.linspace(0, 0., 8)]  # stochastic depth decay rule
        self.ftransformer = nn.Sequential(*[
            Transformer.FTransformerBlock(dim=c_in, drop_path_ratio=dpr1[i])
            for i in range(8)])
        dpr = [x.item() for x in torch.linspace(0, 0., 2)]  # stochastic depth decay rule
        self.transformer = nn.Sequential(*[
            Transformer.TransformerBlock(dim=c_in, num_heads=8, drop_path_ratio=dpr[i])
            for i in range(2)])

        self.pre_head_ln = nn.LayerNorm(c_in)

        # self.conv3d_reduce = nn.Conv3d(
        #     2 * c_in,
        #     c_in,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )

    def forward(self, x):
        # x = self.conv3d_add(x)  # torch.Size([1, 512,16,16,16])
        # print('DMFNet2.py231-MFNet-x4:',x4.shape)
        # ------------Transformer start-----------
        embedding_dim = x.size(1)
        # print('DMFNet2.py235-MFNet-embedding_dim:', embedding_dim)
        dims = [x.size(2), x.size(3), x.size(4)]
        # print('DMFNet2.py235-MFNet-list:', list[2])
        # reshape = [x4.size(2),x4.size(3),x4(4)]
        # print('DMFNet2.py235-MFNet-reshape :', reshape)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # torch.Size([1, 16,16,16,512])
        x = x.view(x.size(0), -1, embedding_dim)  # torch.Size([1, 4096, 512])
        # print('DMFNet2.py238-MFNet-x4 :', x4.shape)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)
        x = self.ftransformer(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)  # torch.Size([1, 4096, 512])
        # print('DMFNet2.py239-MFNet-x4:',x4.shape)
        # resume

        x = myreshape(x, dims)  # torch.Size([1, 512,16,16,16])
        # x = self.conv3d_reduce(x)

        return x

class HPLONet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(HPLONet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        groups = 8
        print("HPLO_Unet2#############"
              ) #[48,96,192,384]

        self.deep_supervision = deep_supervision

        # self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder1 = nn.Conv3d(inplanes, width, kernel_size=3, padding=1, stride=2, bias=False)
        # self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)

        self.encoder2 = nn.Sequential(
            DMFUnit(features[0], features[1], g=groups, stride=2, norm=norm_layer, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(features[1], features[1], g=groups, stride=1, norm=norm_layer, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(features[1], features[1], g=groups, stride=1, norm=norm_layer, dilation=[1, 2, 3])
        )

        # self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder3 = nn.Sequential(
            DMFUnit(features[1], features[2], g=groups, stride=2, norm=norm_layer, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(features[2], features[2], g=groups, stride=1, norm=norm_layer, dilation=[1, 2, 3]),
            DMFUnit(features[2], features[2], g=groups, stride=1, norm=norm_layer, dilation=[1, 2, 3])
        )

        # self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)
        self.encoder4 = nn.Sequential(
            DMFUnit(features[2], features[3], g=groups, stride=2, norm=norm_layer, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(features[3], features[3], g=groups, stride=1, norm=norm_layer, dilation=[1, 2, 3]),
            DMFUnit(features[3], features[3], g=groups, stride=1, norm=norm_layer, dilation=[1, 2, 3])
        )

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)
        self.decoder4 = MFunit(features[3] * 2, features[2], g=groups, stride=1, norm=norm_layer)
        # self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder3 = MFunit(features[2] * 2, features[1], g=groups, stride=1, norm=norm_layer)
        # self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder2 = MFunit(features[1] * 2, features[0], g=groups, stride=1, norm=norm_layer)
        # self.decoder1 = UBlock(features[0], features[0], features[0] // 2, norm_layer, dropout=dropout)
        self.decoder1 = MFunit(features[0] * 2, features[0] // 2, g=groups, stride=1, norm=norm_layer)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        self.ftr = FTrans(384)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        down1 = self.encoder1(x)
        # down2 = self.downsample(down1)

        down2 = self.encoder2(down1)
        # down3 = self.downsample(down2)
        down3 = self.encoder3(down2)
        # down4 = self.downsample(down3)
        down4 = self.encoder4(down3) #384
        # print("down4:",down4.shape)
        # bottom = self.bottom(down4) #384
        bottom = self.ftr(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1)) #192
        # print("bottom:", bottom.shape)
        # print("bottom2:", bottom_2.shape)

        # Decoder

        up3 = self.upsample(bottom_2) #192
        # print(" up3:",  up3.shape)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1)) #96
        # print(" up3:", up3.shape)
        up2 = self.upsample(up3) #96

        # input = torch.rand((2, 192, 64, 64, 64))
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))



        # print("up2:",up2.shape)
        # print("down2:",down2.shape)
        # up2 = self.decoder2(torch.cat([down2, up2], dim=1))

        print("up2:", up2.shape)

        up1 = self.upsample(up2)
        # t = self.test(torch.cat([down1, up1], dim=1))
        # print("t:", t.shape)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        # up1 = self.decoder1(up1)
        seg = self.upsample(up1)
        print("seg:",seg.shape)
        out = self.outconv(seg)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out


