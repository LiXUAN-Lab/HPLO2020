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

class FTrans(nn.Module):
    def __init__(self, c_in = 256):
        super(FTrans, self).__init__()

        self.conv3d_add = nn.Conv3d(
            c_in,
            2 * c_in,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.position_encoding = FixedPositionalEncoding(2 * c_in)
        self.pe_dropout = nn.Dropout(p=0.1)
        dpr1 = [x.item() for x in torch.linspace(0, 0., 8)]  # stochastic depth decay rule
        self.ftransformer = nn.Sequential(*[
            Transformer.FTransformerBlock(dim=2 * c_in, drop_path_ratio=dpr1[i])
            for i in range(8)])
        dpr = [x.item() for x in torch.linspace(0, 0., 2)]  # stochastic depth decay rule
        self.transformer = nn.Sequential(*[
            Transformer.TransformerBlock(dim=2 * c_in, num_heads=8, drop_path_ratio=dpr[i])
            for i in range(2)])

        self.pre_head_ln = nn.LayerNorm(2 * c_in)

        self.conv3d_reduce = nn.Conv3d(
            2 * c_in,
            c_in,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = self.conv3d_add(x)  # torch.Size([1, 512,16,16,16])
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
        # x4 = self.transformer(x4)
        x = self.pre_head_ln(x)  # torch.Size([1, 4096, 512])
        # print('DMFNet2.py239-MFNet-x4:',x4.shape)
        # resume

        x = myreshape(x, dims)  # torch.Size([1, 512,16,16,16])
        x = self.conv3d_reduce(x)

        return x

class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print("unet_features:",features) #[48,96,192,384]

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        # self.ftr = FTrans(384)

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
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        # bottom = self.ftr(bottom)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))
        print("bottom:", bottom.shape)
        print("bottom2:", bottom_2.shape)

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out


class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

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


class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False,  dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

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