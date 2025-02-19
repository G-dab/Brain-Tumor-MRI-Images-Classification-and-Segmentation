import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid

class Conv3D_Block(Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU()
        )

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU()
        )

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(1, kernel, kernel),
                            stride=(1, stride, stride), padding=(0, padding, padding),
                            output_padding=0, bias=True),
            ReLU()
        )

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):
    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)


class UNet3D(Module):
    def __init__(self, num_channels=32, feat_channels=[16, 32, 64, 128, 256], residual='conv'):
        super(UNet3D, self).__init__()

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        self.bn1 = nn.BatchNorm3d(feat_channels[0])
        self.bn2 = nn.BatchNorm3d(feat_channels[1])
        self.bn3 = nn.BatchNorm3d(feat_channels[2])
        self.bn4 = nn.BatchNorm3d(feat_channels[3])
        self.bn5 = nn.BatchNorm3d(feat_channels[4])

        self.bn_d4 = nn.BatchNorm3d(2 * feat_channels[3])
        self.bn_d3 = nn.BatchNorm3d(2 * feat_channels[2])
        self.bn_d2 = nn.BatchNorm3d(2 * feat_channels[1])
        self.bn_d1 = nn.BatchNorm3d(2 * feat_channels[0])

        self.dropout = nn.Dropout(0.5)

        self.one_conv = nn.Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_blk1(x)
        x_low1 = self.pool1(x1)

        x2 = self.conv_blk2(x_low1)
        x_low2 = self.pool2(x2)

        x3 = self.conv_blk3(x_low2)
        x_low3 = self.pool3(x3)

        x4 = self.conv_blk4(x_low3)
        x_low4 = self.pool4(x4)

        base = self.conv_blk5(x_low4)

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.sigmoid(self.one_conv(d_high1))

        return seg

def dice(inputs, targets):
    smooth = 1e-6
    intersection = ((inputs > 0.5).float() * targets).sum()
    
    return (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

# hard dice
def loss_dice(inputs, targets):
    smooth = 1e-6
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = ((inputs > 0.5).float() * targets).sum()
    
    dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return 1 - dice


class Loss(nn.Module):
    def __init__(self, type):
        super(Loss, self).__init__()
        self.type = type

    def forward(self, inputs, targets):
        if self.type == 'dice':
            return loss_dice(inputs, targets)
        elif self.type == 'mixed dice': 
            smooth = 1.
            inputs_flat = inputs.reshape(-1)
            targets_flat = targets.reshape(-1)
            
            intersection = (inputs_flat * targets_flat).sum()
            
            ce = nn.BCEWithLogitsLoss()(inputs, targets.float())
            
            return ce - (2 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        elif self.type == 'mixed dice2': 
            smooth = 1.
            inputs_flat = inputs.reshape(-1)
            targets_flat = targets.reshape(-1)
            
            intersection = (inputs_flat * targets_flat).sum()
            
            ce = nn.BCEWithLogitsLoss()(inputs, targets.float())
            
            return 0.3*ce - (1-0.3)*(2 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)


