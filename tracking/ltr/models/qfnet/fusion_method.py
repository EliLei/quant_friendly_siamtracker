import math

import os
import torch
import torch.nn as nn
from einops import rearrange
from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, relu=True):
        super(ConvBlock, self).__init__()

        if kernel_size!=1:
            self.f = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )
        else:
            self.f = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )

    def forward(self,x):
        return self.f(x)

class Fusion_Concatenate(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_Concatenate, self).__init__()

        self.z_size = size[0]//stride
        self.x_size = size[1]//stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult*n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )


        self.conv_z = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=self.z_size)
        self.conv_x = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=1)

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=2*channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self,z,x):
        # BCHW

        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        z = self.conv_z(z)
        x = self.conv_x(x)

        z = z.expand(-1,-1,self.x_size,self.x_size)
        zx = torch.concat((z,x), dim=1)

        y = self.conv_post(zx)

        return y

class Fusion_PixelWiseAdd(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_PixelWiseAdd, self).__init__()

        self.z_size = size[0] // stride
        self.x_size = size[1] // stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult * n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

        self.conv_z = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=self.z_size)
        self.conv_x = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=1)

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self,z,x):
        # BCHW

        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        z = self.conv_z(z)
        x = self.conv_x(x)

        zx = z+x

        y = self.conv_post(zx)

        return y

class Fusion_FiLM(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_FiLM, self).__init__()

        self.z_size = size[0]//stride
        self.x_size = size[1]//stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult*n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )


        #self.conv_z = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=self.z_size)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_z1 = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv_z2 = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=1)

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self,z,x):
        # BCHW

        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        z = self.avgpool(z)
        z1 = self.conv_z1(z)
        z2 = self.conv_z2(z)

        zx = x*z1 + z2

        y = self.conv_post(zx)

        return y


class Fusion_Correlation(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_Correlation, self).__init__()

        self.z_size = size[0]//stride
        self.x_size = size[1]//stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult * n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

        assert self.z_size%2==1
        self.pad_value = self.z_size//2
        self.pad = nn.ConstantPad2d(self.pad_value, 0.0)

    def forward(self,z,x):
        # BCHW
        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        # 在很多平台上，conv2d的weight和bias不能是动态的，所以不能用
        # B, C = z.shape[:2]
        # z = rearrange(z, 'b c h w -> (b c) h w')
        # z = z.unsqueeze(1)
        # # (B C) 1 H W
        # x = rearrange(x, 'b c h w -> (b c) h w')
        # x = x.unsqueeze(0)
        # # 1 (B C) H W
        # zx = nn.functional.conv2d(x, z, bias=None, groups=B*C, padding=self.z_size//2)
        # zx = zx.squeeze(0)
        # zx = rearrange(zx, '(b c) h w -> b c h w', b=B, c=C)

        # 手动计算卷积
        zx = torch.zeros_like(x)

        x_pad = self.pad(x)
        for offset_i in range(self.pad_value):
            for offset_j in range(self.pad_value):
                zx += x_pad[...,offset_i:offset_i+self.x_size, offset_j:offset_j+self.x_size] * z[...,offset_i:offset_i+1,offset_j:offset_j+1]


        y = self.conv_post(zx)

        return y

class Fusion_Attention(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_Attention, self).__init__()

        self.z_size = size[0]//stride
        self.x_size = size[1]//stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult * n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

        self.pembed_z = nn.Parameter(torch.zeros(1,channels, self.z_size, self.z_size))
        self.pembed_x = nn.Parameter(torch.zeros(1,channels, self.x_size, self.x_size))

        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self,z,x):
        # BCHW
        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        z = z+self.pembed_z
        x = x+self.pembed_x

        B, C, H, W = x.shape


        z = rearrange(z, 'b c h w -> b (h w) c')
        #z = z.reshape(B, C, -1)
        #z = z.permute(0, 2, 1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        #x = x.reshape(B, C, -1)
        #x = x.permute(0, 2, 1)

        zx,_ = self.attn(x, z, z)

        zx = rearrange(zx, 'b (h w) c -> b c h w', h=H, w=W)
        #zx = zx.permute(0, 2, 1)
        #zx= zx.reshape(B, -1, H, W)

        y = self.conv_post(zx)

        return y

class Fusion_GhostAttn(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_GhostAttn, self).__init__()

        self.z_size = size[0] // stride
        self.x_size = size[1] // stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult * n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

        self.pembed_z = nn.Parameter(torch.zeros(1, channels, self.z_size, self.z_size))
        self.pembed_x = nn.Parameter(torch.zeros(1, channels, self.x_size, self.x_size))

        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=channels*2, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self,z,x):

        B, C, H, W = x.shape

        # BCHW
        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        z = z + self.pembed_z
        x = x + self.pembed_x

        B, C, H, W = x.shape

        z = rearrange(z, 'b c h w -> b (h w) c')
        # z = z.reshape(B, C, -1)
        # z = z.permute(0, 2, 1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.reshape(B, C, -1)
        # x = x.permute(0, 2, 1)

        zx, _ = self.attn(x, z, z)
        zx = torch.cat([zx, x], dim=-1)

        zx = rearrange(zx, 'b (h w) c -> b c h w', h=H, w=W)
        # zx = zx.permute(0, 2, 1)
        # zx= zx.reshape(B, -1, H, W)


        y = self.conv_post(zx)

        return y


class Fusion_GhostAttn1(Fusion_GhostAttn):
    def __init__(self, **params):
        super(Fusion_GhostAttn1, self).__init__(**params)

        self.load_pretrain(r'/data/users/leirulin/code/biye/tracking/ltr/checkpoints/ltr/qfnet/qfattn/QFNet_ep0090.pth.tar')

    def load_pretrain(self, pretrain_path):
        if not os.path.exists(pretrain_path):
            #print('pretrain not found')
            return

        ckpt = torch.load(pretrain_path, map_location='cpu')

        statedict = {}

        for k,v in ckpt['net'].items():
            if k.startswith('fusion_method.'):
                statedict[k.lstrip('fusion_method.')]=v

        n_out, n_g, kh, hw = self.conv_post[0].f[0].weight.shape


        with torch.no_grad():

            self.conv_post[0].f[0].weight[:n_out // 2,] = statedict['conv_post.0.f.0.weight']
            self.conv_post[0].f[0].bias[:n_out // 2] = statedict['conv_post.0.f.0.bias']
            self.conv_post[0].f[1].weight[:,:n_out // 2] = statedict['conv_post.0.f.1.weight']
            self.conv_post[0].f[1].bias[:] = statedict['conv_post.0.f.1.bias']

        del statedict['conv_post.0.f.0.weight']
        del statedict['conv_post.0.f.0.bias']
        del statedict['conv_post.0.f.1.weight']
        del statedict['conv_post.0.f.1.bias']
        self.load_state_dict(statedict, strict=False)

        pass

class CornerHead(nn.Module):
    # STARK-ST
    def __init__(self, in_channels=64, feat_sz=15, stride=16):
        super(CornerHead, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        self.conv1_tl = ConvBlock(in_channels=in_channels, out_channels=in_channels // 2)
        self.conv2_tl = ConvBlock(in_channels=in_channels//2, out_channels=in_channels // 4)
        self.conv3_tl = ConvBlock(in_channels=in_channels//4, out_channels=in_channels // 8)
        self.conv4_tl = ConvBlock(in_channels=in_channels//8, out_channels=in_channels // 16)
        self.conv5_tl = ConvBlock(in_channels=in_channels//16, out_channels=1, bias=True)

        self.conv1_br = ConvBlock(in_channels=in_channels, out_channels=in_channels // 2)
        self.conv2_br = ConvBlock(in_channels=in_channels // 2, out_channels=in_channels // 4)
        self.conv3_br = ConvBlock(in_channels=in_channels // 4, out_channels=in_channels // 8)
        self.conv4_br = ConvBlock(in_channels=in_channels // 8, out_channels=in_channels // 16)
        self.conv5_br = ConvBlock(in_channels=in_channels // 16, out_channels=1)

        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        #from matplotlib import pyplot as plt;plt.figure('tl');plt.imshow(score_map_tl.reshape(score_map_tl.shape[-2:]).detach().cpu().numpy());plt.figure('br');plt.imshow(score_map_br.reshape(score_map_br.shape[-2:]).detach().cpu().numpy())
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

class ClsHead(nn.Module):
    def __init__(self, in_channels=64, feat_sz=15):
        super(ClsHead, self).__init__()

        linear_channels = feat_sz*feat_sz*(in_channels//16)

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(in_channels=in_channels//4, out_channels=in_channels // 16, kernel_size=3, padding=1)

        self.linear = nn.Linear(in_features=linear_channels, out_features=1)

    def forward(self, zx):
        zx = self.conv1(zx)
        zx = self.conv2(zx)

        # B C H W
        zx = rearrange(zx, 'B C H W -> B (H W C)')

        zx = self.linear(zx)

        return zx

class Fusion_GhostAttn2x(nn.Module):
    def __init__(self, size=(112,240), width_mult=1.0, n_channels=128, in_channels=256 ,out_channels=256, stride=16):
        super(Fusion_GhostAttn2x, self).__init__()

        self.z_size = size[0] // stride
        self.x_size = size[1] // stride

        self.width_mult = width_mult
        self.in_channles = in_channels
        self.out_channles = out_channels

        channels = int(width_mult * n_channels)

        self.pre_conv_z = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
        self.pre_conv_x = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

        self.pembed_z = nn.Parameter(torch.zeros(1, self.z_size*self.z_size, 1))
        self.pembed_x = nn.Parameter(torch.zeros(1, self.x_size*self.x_size, 1))

        self.attn1 = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=channels*2, out_features=channels, bias=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)

        self.conv_post = nn.Sequential(
            ConvBlock(in_channels=channels*2, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self,z,x):

        B, C, H, W = x.shape

        # BCHW
        z = self.pre_conv_z(z)
        x = self.pre_conv_x(x)

        B, C, H, W = x.shape

        z = rearrange(z, 'b c h w -> b (h w) c')
        # z = z.reshape(B, C, -1)
        # z = z.permute(0, 2, 1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.reshape(B, C, -1)
        # x = x.permute(0, 2, 1)
        z = z + self.pembed_z
        x = x + self.pembed_x

        z_x0 = torch.cat([z,x],dim=1)

        z_x1, _ = self.attn1(z_x0, z, z)

        z_x1 = torch.cat([z_x1, z_x0], dim=-1)

        z_x1 = self.fc1(z_x1)

        z1 = z_x1[:,:self.z_size*self.z_size,:]
        x1 = z_x1[:,-self.x_size*self.x_size:,:]

        z1 = z1 + self.pembed_z
        x1 = x1 + self.pembed_x

        zx, _ = self.attn2(x1, z1, z1)

        zx = torch.cat([zx, x1], dim=-1)

        zx = rearrange(zx, 'b (h w) c -> b c h w', h=H, w=W)
        # zx = zx.permute(0, 2, 1)
        # zx= zx.reshape(B, -1, H, W)


        y = self.conv_post(zx)

        return y



class BBoxHead(nn.Module):
    def __init__(self, in_channels=64, feat_sz=15, stride=16):
        super(BBoxHead, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        mlp_channels = feat_sz*feat_sz*in_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_features=mlp_channels, out_features=mlp_channels // 64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_channels // 64, out_features=mlp_channels // 320),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_channels // 320, out_features=4),
        )

    def forward(self, x):
        # B C H W
        x = rearrange(x, "B C H W -> B (C H W)")
        x = self.mlp(x)
        return x


if __name__ == '__main__':

    nchannel = 68
    out_channels = 64
    batch = 1
    imgsz_z = 112
    imgsz_x = 240
    stride = 16
    featsz_z = imgsz_z//stride
    featsz_x = imgsz_x//stride

    shape1 = (batch, nchannel, featsz_z, featsz_z)
    shape2 = (batch, nchannel, featsz_x, featsz_x)

    z = torch.randn(shape1)
    x = torch.randn(shape2)

    f1 = Fusion_Concatenate(width_mult=0.9, in_channels=nchannel, out_channels=out_channels, n_channels=128)
    f2 = Fusion_PixelWiseAdd(width_mult=1.0, in_channels=nchannel, out_channels=out_channels, n_channels=128)
    f3 = Fusion_Correlation(width_mult=1.1, in_channels=nchannel, out_channels=out_channels, n_channels=128)
    f4 = Fusion_Attention(width_mult=1.1, in_channels=nchannel, out_channels=out_channels, n_channels=128)
    f5 = Fusion_GhostAttn(width_mult=1.1, in_channels=nchannel, out_channels=out_channels, n_channels=128)
    f1.eval()
    f2.eval()
    f3.eval()
    f4.eval()
    f5.eval()

    y1 = f1(z, x)
    y2 = f2(z, x)
    y3 = f3(z, x)
    y4 = f4(z, x)
    y5 = f5(z, x)

    print(y1.shape,y2.shape,y3.shape,y4.shape,y5.shape)

    summary(f1, input_size=(shape1, shape2), device='cuda')
    summary(f2, input_size=(shape1, shape2), device='cuda')
    summary(f3, input_size=(shape1, shape2), device='cuda')
    summary(f4, input_size=(shape1, shape2), device='cuda')
    summary(f5, input_size=(shape1, shape2), device='cuda')

    h1 = CornerHead(in_channels=out_channels, feat_sz=featsz_x)
    h2 = ClsHead(in_channels=out_channels, feat_sz=featsz_x)

    summary(h1, input_size=y1.shape, device='cuda')
    summary(h2, input_size=y1.shape, device='cuda')


    pass