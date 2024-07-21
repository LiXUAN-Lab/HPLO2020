import torch.nn as nn
import torch.nn.functional as F
import torch
from .PositionalEncoding import FixedPositionalEncoding
from . import Transformer

try:
    from .sync_batchnorm import batchnorm
except:
    pass
# from sync_batchnorm import batchnorm

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
def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = batchnorm.SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

# 1x1x1
class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.gn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.gn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
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

class Encoder(nn.Module):
    def __init__(self, c=4,n=32,channels=128, groups=16,norm='gn'):
        super(Encoder, self).__init__()
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)
        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

    def forward(self, x):
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)

        return x1, x2, x3
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
        x = self.transformer(x)
        x = self.pre_head_ln(x)  # torch.Size([1, 4096, 512])
        # print('DMFNet2.py239-MFNet-x4:',x4.shape)
        # resume

        x = myreshape(x, dims)  # torch.Size([1, 512,16,16,16])
        x = self.conv3d_reduce(x)

        return x

class HPLONet(nn.Module):  # softmax

    def __init__(self, c=4, n=32, channels=128, groups=16, norm='gn', num_classes=4):
        super(HPLONet, self).__init__()
        print("HPLO###############")
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        # self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, stride=1, bias=True)


        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.encoder = Encoder(c, n, channels, groups, norm)
        self.ftr = FTrans()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        x4 = self.ftr(x3)

        y1 = torch.cat([x3, x4], dim=1)

        # y1 = torch.cat([x3, x4], dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)  # H//4

        y2 = torch.cat([x2, y2], dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)  # H//2

        y3 = torch.cat([x1, y3], dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)  # torch.Size([1, 32, 128, 128, 128])

        y4 = self.seg(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4
