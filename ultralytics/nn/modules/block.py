# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "Concat2",
    "ADD",
    "SimAM",
    "ShuffleAttention",
    "GAM_Attention",
    "CBAM2",
    "CoordAtt",
    "ECA",
    "SEAttention",
    "GLCBAM",
    "S2Attention",
    "SKAttention",
    "GLF",
    "NAM",
    "GCBAM",
    "SACBAM",
    "MdC2f",
    "C2f_Invo"
)
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.SiLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, n=3,stride=1, scale=0.1, map_reduce=4):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=1),
            BasicConv(2*inter_planes, 2*inter_planes , kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2*inter_planes, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
        )



    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out
    
class C2f_FEM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList([*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n//2)),FEM(self.c,self.c)] )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
import numpy as np
import torch
from torch import nn
from torch.nn import init

# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x


def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) 
        a=torch.sum(torch.sum(x_all,1),1) 
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) 
        hat_a=hat_a.reshape(b,self.k,c) 
        bar_a=self.softmax(hat_a) 
        attention=bar_a.unsqueeze(-2) 
        out=attention*x_all 
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out
#NAM
class NAM(nn.Module):
    def __init__(self, channels,c2, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.conv=Conv(channels,c2,1,1)
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
    def forward(self, x):
        x=torch.cat(x,1)
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        x=self.conv(x)
        return x
    
    
class GLF(nn.Module):

    def __init__(self, c1,c2,channel=512, reduction=16):
        super().__init__()
        channel=c1
        self.conv=Conv(c1,c2,1,1)
        self.d=1

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #全局池化
        # 全局特征提取
        self.fc1 = nn.Sequential(
         
            nn.Conv2d(channel, channel // reduction,1,1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1,1),
            nn.BatchNorm2d(channel ),
            nn.Sigmoid()
        )
        # 局部特征提取
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,1,1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1,1),
            nn.BatchNorm2d(channel),
        )



    def forward(self, x):
        x=torch.cat(x, self.d)
        b, c, _, _ = x.size()
        
        # 全局特征mul
        y = self.avg_pool(x)
        y = self.fc1(y).view(b, c, 1, 1)

        #局部特征
        y1= self.fc2(x)

        x=x * y.expand_as(x) 
        #局部特征add
        x=torch.add(x, y1)
        
        x=self.conv(x)

        return x
    




from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self,c1,c2, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):

        x=torch.cat(x,1)
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        V=self.conv(V)
        return V


    


class S2Attention(nn.Module):

    def __init__(self, c1,c2,channels=512 ):
        super().__init__()
        channels=c1
        self.conv=Conv(c1,c2,1,1)

        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(c1)

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b,c,w,h = x.size()
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        x=self.conv(x)
        return x
  
  

 
 
###################### EffectiveSE     ####     end   by  AI&CV  ###############################

import numpy as np
import torch
from torch import nn
from torch.nn import init

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        #self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out

class CBAM2(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM2, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x=torch.cat(x, self.d) 
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        x=self.conv(out)
        return x


class CSFM(nn.Module):
    def __init__(self, c1,c2):
        super(CSFM, self).__init__()
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        _,c,_,_=x[0].shape
        x3=x[0]
        x4=x[1]
        x=torch.cat(x, self.d) 
        out = self.channel_attention(x) * x
        x1, x2 = torch.split(out, c, dim =self.d)

        x1=x1*x3
        x2=x2*x4
        # x1+=x[0]
        # x2+=x[1]
        out=torch.add(x1,x2)
        # out = self.spatial_attention(out) * out
        
        return out

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output
    



class SACBAM(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        


        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)

        x = self.channel_shuffle(x, 2)
        x_channel=self.channel_attention(x) * x
        out=self.spatial_attention(x_channel) * x_channel
        out=self.conv(out)
        return out
    


    
class GCBAM(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        


        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b, c, h, w = x.size()
        # group into subfeatures

        # x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        # x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # # channel attention
        # x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        # x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        # x_channel = x_0 * self.sigmoid(x_channel)

        # # spatial attention
        # x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        # x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        # x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        x_channel=self.channel_attention(x) * x
        
        out=self.spatial_attention(x_channel) * x_channel
        # concatenate along channel axis
        # out = torch.cat([x_channel, x_spatial], dim=1) 
        # out = out.contiguous().view(b, -1, h, w)
        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out=self.conv(out)
        return out
    
    
# 局部CBAM
class GLCBAM(nn.Module):
    def __init__(self, c1,c2):
        super(GLCBAM, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
        mid_channel=c1//16
        
        #局部特征
        self.localConv = nn.Sequential(          
            nn.Conv2d(in_channels=c1, out_channels=mid_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=c1,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        x=torch.cat(x, self.d) 
        y=x
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        
        local=self.localConv(y)
        out=torch.add(local,out)
        
        x=self.conv(out)

        return x


class SACBAM(nn.Module):
    def __init__(self, c1,c2):
        super(SACBAM, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
        self.SA=ShuffleAttention(c1,c2)
        mid_channel=c1//16
        
 

    def forward(self, x):
        x=torch.cat(x, self.d) 
        y=x
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        
        local=self.SA(y)
        out=torch.add(local,out)
    
        x=self.conv(out)

        return x
    

class SEAttention(nn.Module):

    def __init__(self, c1,c2,channel=512, reduction=16):
        super().__init__()
        channel=c1
        # self.conv=Conv(c1,c2,1,1)
        self.d=1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x=x * y.expand_as(x) 
        # x=self.conv(x)
        return x
    
class Concat2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,c2,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        #self.conv=nn.Conv2d(c1,c2,1,1,bias=False)
        self.conv=Conv(c1,c2,1,1)

    def forward(self, x):
        x=torch.cat(x, self.d)
        x=self.conv(x)
        return x
class SA(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out
    
from torch.nn import init
from torch.nn.parameter import Parameter

class SimAM(torch.nn.Module):
    def __init__(self, c1,c2,e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.d=1
        self.conv=Conv(c1,c2,1,1)

        

    def forward(self, x):
        x=torch.cat(x, self.d)
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        x= x * self.activaton(y)
        x=self.conv(x)
        return x




import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp,c2, reduction=32):
        super(CoordAtt, self).__init__()
        self.conv=Conv(inp,c2,1,1)
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
 
    def forward(self, x):
        x=torch.cat(x,dim=1)
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return self.conv(out)

import torch
from torch import nn
from torch.nn.parameter import Parameter
class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x
    
# class ECA(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, c1,c2, k_size=3):
#         super(ECA, self).__init__()
#         self.conv1=Conv(c1,c2,1,1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         x=torch.cat(x,dim=1)
#         y = self.avg_pool(x)
 
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
 
#         return self.conv1(x * y.expand_as(x))
    
class GAM_Attention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()
        self.conv=Conv(c1,c2,1,1)

        c2=c1
        self.d=1
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        x=torch.cat(x,dim=self.d)
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        out=self.conv(out)
        return out
    
class ShuffleAttention(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16, G=8):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out=self.conv(out)
        return out






        
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        # a 8400
        # c1=16 
        # 4 16 a 
        # 16 4 a 
        # 4 a
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class MdC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Md(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,deiltations=i+1) for i in range(n))
        

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class CDC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        # 3 1 3 8 5 2  5 2     k= 3, 3, 5, and 5 and d= 1, 8, 2, and 3
        if n==1:
           # high pass d
           # 3 1 /3 8/5 3
           self.m = nn.ModuleList(Md(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,deiltations=8))


        else :
           # low pass c
           # 3 1/3 8/ 5 2/ 5 3/ 3 3/ 5 5 
           self.m = nn.ModuleList((Md(self.c, self.c, shortcut, g, k=((3, 3), (5, 5)), e=1.0,deiltations=2),
                                  Md(self.c, self.c, shortcut, g, k=((3, 3), (5, 5)), e=1.0,deiltations=3)) )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    


class C2f_F(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = c1//4 # hidden channels
        self.c1=self.c*3
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c,self.c,k=3,s=1) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x=self.cv1(x)
        c1=3*self.c
        x1 = x[:, :c1, :, :]  
        # 第二部分  
        x2 = x[:, c1:, :, :] 
        y=list([x1,x2])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


import torch
import torch.nn as nn
from torch.nn import functional as F
 
 

 
class Involution(nn.Module):
 
    def __init__(self, c1, c2, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = Conv(
            c1, c1 // reduction_ratio, 1)
        self.conv2 = Conv(
            c1 // reduction_ratio,
            kernel_size ** 2 * self.groups,
            1, 1)
 
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
 
    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)
 
        return out

from ultralytics.utils.torch_utils import make_divisible


class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor
    

class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)
        
        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)
        
        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None
        
        self.add_identity = add_identity and inc == ouc
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)
        
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x
    


class C2f_PKIModule(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        self.m = nn.ModuleList(PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n))

class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride

        branch_features = oup // 2 # 输出的一半
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # copy input
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            #Dw卷积
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            #Pw
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out
    
class C2f_Shufflenet(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ShuffleNetV2(self.c, self.c,1) for _ in range(n))

class C2f_Invo(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(InvoConv(self.c, self.c,1) for _ in range(n))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class InvoConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Involution(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class Md(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,deiltations=1):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g,d=deiltations)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)  # [bs, gc] -> [bs, ec]
        guide = guide.view(bs, -1, self.nh, self.hc)  # [bs, ec] -> [bs, 1, nh, hc]
        embed = self.ec(x) if self.ec is not None else x  # [bs, c1, h, w] -> [bs, ec, h, w]
        embed = embed.view(bs, self.nh, self.hc, h, w)  # [bs, ec, h, w] -> [bs, nh, hc, h, w]

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)  # 点积计算相似度
        aw = aw.max(dim=-1)[0]  # 沿最后一个维度取最大值
        aw = aw / (self.hc ** 0.5)  # 缩放
        aw = aw + self.bias[None, :, None, None]  # 添加偏置
        aw = aw.sigmoid() * self.scale  # Sigmoid激活并缩放

        x = self.proj_conv(x)  # [bs, c1, h, w] -> [bs, c2, h, w]
        x = x.view(bs, self.nh, -1, h, w)  # [bs, c2, h, w] -> [bs, nh, hc, h, w]
        x = x * aw.unsqueeze(2)  # 应用注意力权重
        return x.view(bs, -1, h, w)  # [bs, nh, hc, h, w] -> [bs, c2, h, w]


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

 
class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

# https://mp.weixin.qq.com/s/26H0PgN5sikD1MoSkIBJzg
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = Conv(in_features, filters, act=False)
         self.c1 = Conv(filters, filters, 3)
         self.c2 = Conv(filters, filters, 3)
         self.c3 = Conv(filters, filters, 3)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.drop = nn.Dropout2d(0.1)
         self.bn1 = nn.BatchNorm2d(filters)
         self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x


class C2f_PPA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PPA(self.c, self.c) for _ in range(n))

from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # x = x.clone()  # !!! Keep the original input intact for the residual connection later
        # x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        # return x
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x



class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()
    

    def forward(self, x):
        

        x1 = self.primary_conv(x)  # mg
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn   # 激活函数
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
 
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)



class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )
          

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x) #
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)
    

class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy
        # 1x1 conv + bn + SiLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv) #indentity
        self.fusion_bn = nn.Sequential(*fusion_bn) #fusion bn

        # dwconv BN Silu
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)  # conv1x1 SiLu
        x2 = self.cheap_operation(x1) # dw BN SiLu
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))# indentity x1 + bn
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        mid_chs=in_chs//2
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)

class C2f_RG(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepGhostBottleneck(self.c, self.c) for _ in range(n))




### bifpn##


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU()(self.shuf(x2))

class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class GSBottleneckns(GSBottleneck):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__(c1, c2, k, s, e)
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConvns(c1, c_, 1, 1),
            GSConvns(c_, c2, 3, 1, act=False))
        
class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)

class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

class VoVGSCSPns(VoVGSCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))

class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)


class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans
    







class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)
        
###### bifpn###


 

class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()

        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo
class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,c2,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        self.Fm=Fusion_module(channels=c2)


    def forward(self, x):
        # x1=self.conv1(x[0])
        # x2=self.conv2(x[1])

        x=self.Fm(x[0],x[1])
        # x=torch.cat([x1,x2], self.d)

        return x
################空###################

class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=1,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            # nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        # self.adjust_channel = None
        # if inc != dim:
        #     self.adjust_channel = Conv(inc, dim, 1)

        # if layer_scale_init_value > 0:
        #     self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #     self.forward = self.forward_layer_scale
        # else:
        #     self.forward = self.forward

    def forward(self, x):
        # if self.adjust_channel is not None:
        #     x = self.adjust_channel(x)
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(self.mlp(x))
        
        return self.mlp(x)

    def forward_layer_scale(self, x):
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(
        #     self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C2f_Faster(C2f):
    """简化c2f模块,使用Faster_Block替代Bottleneck模块"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))


class ADD5(nn.Module):
    #  Add two tensors

    def __init__(self, args):
        super(ADD, self).__init__()
        # 128 256 512
        self.channels = args
        self.fusion_gate = nn.Sequential(
            Conv(2 * self.channels, self.channels, 1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb = x[0]
        ir = x[1]
        gate = self.fusion_gate(torch.cat([rgb, ir], dim=1))
        fused = gate * rgb + (1 - gate) * ir
        return rgb + self.gamma * fused
    

class RIFusion(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,r=16,dimension=1):
        super().__init__()

    def forward(self, x):
        return x

    
# class RIFusion(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, c1,r=16,dimension=1):
#         super().__init__()
#         self.c1=c1*2
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(self.c1, self.c1 // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.c1 // r, self.c1, bias=False),
#             nn.Sigmoid()
#             # nn.Sigmoid(inplace=True)
#         )
#     def forward(self, x):
#         # return x
#         b, _, _, _ = x.size()
#         y = self.avg_pool(x).view(b, self.c1)
#         y = self.fc(y).view(b, self.c1, 1, 1)

#         x1=x*y
#         return x+torch.cat((x1[:,self.c1//2:,...],x1[:,:self.c1//2,...]),dim=1)



    

# class CAFusion(nn.Module):
#     """跨模态注意力融合模块CrossModalAttentionFusion"""

#     def __init__(self, channels, reduction=4, fusion_method='gate'):
#         super().__init__()
#         #投影层
#         self.query_proj_rgb = Conv(channels, channels // reduction, 1)
#         self.key_proj_ir = Conv(channels, channels // reduction, 1)
#         self.value_proj_ir = Conv(channels, channels, 1)

#         self.query_proj_ir = Conv(channels, channels // reduction, 1)
#         self.key_proj_rgb = Conv(channels, channels // reduction, 1)
#         self.value_proj_rgb = Conv(channels, channels, 1)


#     def forward(self, rgb, ir):
        
#         # RGB → IR 注意力 (RGB从IR获取信息)
#         q_rgb = self.query_proj_rgb(rgb)
#         k_ir = self.key_proj_ir(ir)
#         v_ir = self.value_proj_ir(ir)
#         rgb_enhanced_by_ir = self._attention(q_rgb, k_ir, v_ir)

#         # IR → RGB 注意力 (IR从RGB获取信息)
#         q_ir = self.query_proj_ir(ir)
#         k_rgb = self.key_proj_rgb(rgb)
#         v_rgb = self.value_proj_rgb(rgb)
#         ir_enhanced_by_rgb = self._attention(q_ir, k_rgb, v_rgb)

#         # 融合策略1: 残差连接 (最常用)
#         rgb_out = rgb + rgb_enhanced_by_ir
#         ir_out = ir + ir_enhanced_by_rgb

#         # # 融合策略2: 拼接后卷积融合
#         # fused = torch.cat([rgb_enhanced_by_ir, ir_enhanced_by_rgb], dim=1)
#         # rgb_out = self.fuse_conv(fused)
#         # ir_out = ir_enhanced_by_rgb  # 或者其他处理方式

#         return rgb_out, ir_out
#         # return rgb, ir

#     def _attention(self, q, k, v):
#         """标准注意力计算"""
#         B, C, H, W = q.shape
#         q = q.view(B, -1, H * W).permute(0, 2, 1)
#         k = k.view(B, -1, H * W)
#         v = v.view(B, -1, H * W).permute(0, 2, 1)

#         attn = F.softmax(q @ k / (C ** 0.5), dim=-1)
#         out = (attn @ v).permute(0, 2, 1).view(B, -1, H, W)
#         return out
    

class ADD56(nn.Module):
    #  Add two tensors

    def __init__(self, args):
        super(ADD, self).__init__()
        # 128 256 512
        self.channels = args
        # self.fusion_gate = nn.Sequential(
        #     Conv(2 * self.channels, self.channels, 1),
        #     nn.Sigmoid()
        # )
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # rgb = x[0]
        # ir = x[1]
        # gate = self.fusion_gate(torch.cat([rgb, ir], dim=1))
        # fused = gate * rgb + (1 - gate) * ir
        # return rgb + self.gamma * fused
        return torch.add(x[0], x[1])
    
class DECA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        """convolution pyramid"""
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, int(self.kernel_size/kernel1/kernel2), int(self.kernel_size/kernel1/kernel2), 0,
                      groups=channel),
            nn.SiLU()
        )
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        # print(f"ADD: {x.shape}")
        b, c, h, w = x[0].size()
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t))) if min(h, w) >= self.kernel_size else torch.mean(
                                                                                    glob_t, dim=[2, 3], keepdim=True)
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return self.act2(result_vi + result_ir)


class FrequencyCrossAttention_1(nn.Module):
    """
    基于频域的交叉注意力模块 (Frequency-Domain Cross-Attention)
    利用 FFT 实现全局、通道级的 QKV 交互。
    """
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        
        # 1. 频域降维/投影层 (用于生成 Q, K, V)
        # 注意：这里的卷积在空间域进行，但用于准备频域的 QKV
        hidden_channels = channels // reduction_ratio
        
        # 定义投影层：将 RGB 和 IR 的特征通道数降低
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 定义输出层：将融合结果恢复通道数
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            F_rgb (torch.Tensor): RGB 特征图 (B, C, H, W)
            F_ir (torch.Tensor): IR 特征图 (B, C, H, W)
        """
        F_rgb, F_ir = x[0],x[1]

        B, C, H, W = F_rgb.shape
        
        # 0. 投影到隐通道，减少 FFT 后的计算量
        # Q (Query) 来自 RGB， K, V (Key, Value) 来自 IR (作为示例)
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        V_ir = self.proj_ir(F_ir) # 可以使用独立的 V 投影，这里简化
        
        H_hidden = Q_rgb.shape[2]
        W_hidden = Q_rgb.shape[3]
        C_hidden = Q_rgb.shape[1]
        
        # 1. 快速傅里叶变换 (RFFT) 到频域
        # 输出形状: [B, C_hidden, H_hidden, W_hidden//2 + 1] (复数)
        Q_freq = torch.fft.rfft2(Q_rgb, norm='backward')
        K_freq = torch.fft.rfft2(K_ir, norm='backward')
        V_freq = torch.fft.rfft2(V_ir, norm='backward')
        
        # 2. 频域注意力计算 (核心步骤)
        
        # 将复数张量 Q, K, V 转换为实部和虚部堆叠的实数张量
        # 形状: [B, C_hidden, H, W//2 + 1] -> [B, 2*C_hidden, H, W//2 + 1]
        Q_complex = torch.view_as_real(Q_freq).permute(0, 1, 4, 2, 3).reshape(B, C_hidden * 2, H_hidden, W_hidden // 2 + 1)
        K_complex = torch.view_as_real(K_freq).permute(0, 1, 4, 2, 3).reshape(B, C_hidden * 2, H_hidden, W_hidden // 2 + 1)
        V_complex = torch.view_as_real(V_freq).permute(0, 1, 4, 2, 3).reshape(B, C_hidden * 2, H_hidden, W_hidden // 2 + 1)

        # 全局上下文建模 (GAP)
        # 由于我们希望它是通道级的注意力，我们对空间维度进行平均

        # 1. 对空间维度取平均，得到形状 [B, 2*C_hidden, 1, 1]
        Q_context_pooled = Q_complex.mean(dim=[2, 3], keepdim=True)
        K_context_pooled = K_complex.mean(dim=[2, 3], keepdim=True)
        
        # 2. 移除后面两个为 1 的维度，但保留 Batch 维度 0
        Q_context = Q_context_pooled.flatten(start_dim=1) # 形状: [B, 2*C_hidden]
        K_context = K_context_pooled.flatten(start_dim=1) # 形状: [B, 2*C_hidden]
        
        # 3. 计算通道级注意力分数 (Score = Q dot K)
        # 现在 Q_context 和 K_context 至少是 [B, C'] 形状，
        # dim=1 始终有效 (代表通道维度)

        # 计算通道级注意力分数 (Score = Q dot K)
        # 注意力分数 (A) 形状: [B, 1] (这里是通道级，简化为标量)
        # 实际操作中，通常在通道维度上进行矩阵乘法，这里使用点积的通道平均简化
        
        # 通道级点积注意力 (Simplicity over matrix multiplication)
        A_raw = torch.sum(Q_context * K_context, dim=1, keepdim=True) / (C_hidden * 2) ** 0.5
        
        # Sigmoid 激活得到注意力权重 (W_att)
        W_att = torch.sigmoid(A_raw) # [B, 1]

        # 3. 应用注意力权重到 V (Value)
        # W_att 形状扩展到频域 V 的形状: [B, 1] -> [B, 2*C_hidden, H, W//2 + 1]
        W_att_exp = W_att.view(B, 1, 1, 1).expand_as(V_complex)
        
        # 频域加权
        F_attended_complex = V_complex * W_att_exp
        
        # 4. 逆变换准备 (恢复复数)
        # 恢复形状: [B, C_hidden, 2, H, W//2 + 1]
        F_attended_complex_permuted = F_attended_complex.reshape(B, C_hidden, 2, H_hidden, W_hidden // 2 + 1).permute(0, 1, 3, 4, 2)

        # 关键修复：在 view_as_complex 之前，强制张量连续
        F_attended_complex_final = F_attended_complex_permuted.contiguous()
        
        # 恢复为 PyTorch 复数
        F_attended_freq = torch.view_as_complex(F_attended_complex_final)

        # 5. 逆傅里叶变换 (IRFFT)
        F_attended_spatial = torch.fft.irfft2(F_attended_freq, s=(H_hidden, W_hidden), norm='backward')
        
        # 6. 最终融合与残差连接
        # 将融合结果恢复到原始通道数 C，并加入残差连接
        F_out = self.output_conv(F_attended_spatial)
        
        return F_out + F_rgb # 使用 RGB 作为残差连接的基础


class FrequencyCrossAttention(nn.Module):
    """
    改进版频域交叉注意力模块 (Frequency-Domain Cross-Attention v2)
    特点：
    1. 振幅与相位解耦融合：利用 IR 的振幅信息增强 RGB 特征。
    2. 通道级频域权重：学习每个通道在频域的重要性。
    3. 保持复数运算：确保频域物理意义完整。
    """
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        # 投影层
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 频域通道注意力权重生成器 (针对振幅)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        """
        Args:
            x[0]: F_rgb (B, C, H, W)
            x[1]: F_ir (B, C, H, W)
        """
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # 1. 投影到隐空间
        Q_rgb = self.proj_rgb(F_rgb) # (B, C_h, H, W)
        K_ir = self.proj_ir(F_ir)   # (B, C_h, H, W)

        # 2. FFT 变换到频域
        # rfft2 得到的形状: (B, C_h, H, W//2 + 1)
        rgb_freq = torch.fft.rfft2(Q_rgb, norm='backward')
        ir_freq = torch.fft.rfft2(K_ir, norm='backward')

        # 3. 振幅与相位提取
        rgb_amp = torch.abs(rgb_freq)
        rgb_phase = torch.angle(rgb_freq)
        
        ir_amp = torch.abs(ir_freq)
        # ir_phase = torch.angle(ir_freq) # 如果需要 IR 结构可以用

        # 4. 交叉注意力建模
        # 这里我们计算 IR 振幅对 RGB 振幅的调制权重
        # 将频域振幅视为一种特殊的特征图进行通道注意力计算
        att_weight = self.freq_gate(ir_amp) 
        
        # 融合振幅：结合 RGB 原有能量和被 IR 调制的能量
        # 这种做法能突出 IR 中的显著目标，同时保留 RGB 的背景
        enhanced_amp = rgb_amp * att_weight + ir_amp * (1 - att_weight)

        # 5. 频域恢复（重构复数）
        # 使用 polar(振幅, 相位) 重新合成，保留 RGB 的相位（结构信息）
        combined_freq = torch.polar(enhanced_amp, rgb_phase)

        # 6. 逆 FFT 变换回空间域
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        # 7. 后处理与残差
        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # 残差连接：建议同时考虑 RGB 和 IR 的信息流
        return out + F_rgb + F_ir



class FrequencyCrossAttention_V3(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        # 投影层
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 光照感知模块：判断当前环境是偏向 RGB 还是 IR
        self.illum_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 2. 频域空间注意力：学习频谱中哪些位置（频率）更重要
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照感知权重 ---
        # 动态决定对 RGB 和 IR 的整体信任度
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1)) # (B, 1, 1, 1)

        # --- B. 频域交互 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)

        rgb_freq = torch.fft.rfft2(Q_rgb, norm='backward') 
        ir_freq = torch.fft.rfft2(K_ir, norm='backward')

        # 提取振幅和相位
        amp_rgb, pha_rgb = torch.abs(rgb_freq), torch.angle(rgb_freq)
        amp_ir, pha_ir = torch.abs(ir_freq), torch.angle(ir_freq)

        # --- C. 核心改进：振幅与相位的协同增强 ---
        # 1. 学习频域的显著性掩码 (Focus on important frequencies)
        freq_mask = self.spatial_gate(amp_ir)
        
        # 2. 振幅融合 (考虑光照感知的残差混合)
        # 如果 illum 接近 1 (白天)，倾向于保留更多 RGB 细节
        mixed_amp = illum * amp_rgb + (1 - illum) * amp_ir
        enhanced_amp = mixed_amp * freq_mask + amp_ir * (1 - freq_mask)

        # 3. 相位融合 (创新点：不再只用 RGB 相位)
        # 在夜间 (illum 小)，引入一部分 IR 相位来校正车辆轮廓
        # 使用线性插值处理相位（注意：相位插值需谨慎，这里用简单的线性权重作为一种探索）
        mixed_phase = illum * pha_rgb + (1 - illum) * pha_ir

        # --- D. 恢复 ---
        combined_freq = torch.polar(enhanced_amp, mixed_phase)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # 增加跨模态残差
        return out + F_rgb * illum + F_ir * (1 - illum)
    

class FrequencyCrossAttention_V4(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        # 投影层
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 改进：像素级光照感知 (不再池化到 1x1)
        # 能够处理 "车灯处信RGB，黑暗处信IR" 的局部情况
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 2. 频域空间注意力 (保持不变，用于提取高频边缘)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 像素级光照权重 (Spatial Adaptive Weighting) ---
        # 输出尺寸: (B, 1, H, W)
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # --- B. 频域变换 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)

        # rfft2 得到复数张量
        rgb_freq = torch.fft.rfft2(Q_rgb, norm='backward')
        ir_freq = torch.fft.rfft2(K_ir, norm='backward')

        # --- C. 改进：复数域融合 + 频域注意力 ---
        
        # 1. 振幅注意力：我们仍然希望基于 IR 的纹理来增强频率响应
        amp_ir = torch.abs(ir_freq)
        freq_mask = self.spatial_gate(amp_ir)
        
        # 2. 复数加权融合 (这是数学上更严谨的融合方式)
        # 在复数域融合自动处理了振幅和相位的关系
        # illum 广播到频域尺寸需要注意，频域通常 W 减半，这里我们在空间域加权可能更好，
        # 但为了保持频域特性，我们采用如下策略：
        
        # 策略：直接混合复数向量
        # illum 是空间域权重，我们需要它作用在空间域或者近似看作低频权重。
        # 为了简化并保留你的设计初衷，我们回到振幅/相位，但修正相位融合逻辑。
        
        # --- 修正后的 V3 逻辑 ---
        amp_rgb = torch.abs(rgb_freq)
        pha_rgb = torch.angle(rgb_freq)
        pha_ir = torch.angle(ir_freq)
        
        # 振幅融合 (使用像素级 illum)
        # 注意: illum 是 (B,1,H,W)，在频域操作需要调整尺寸或视为一致。
        # 为了避免频域尺寸不匹配，我们先在空间域融合特征，再做频域增强
        
        # === 替代方案：更稳健的频域增强路径 ===
        # 1. 基础融合特征 (空间域加权)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # 2. 转频域
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # 3. 注入 IR 的高频/边缘信息 (通过 mask)
        # 如果 IR 的某个频率响应很强(边缘)，我们要增强它
        enhanced_amp = fused_amp + (fused_amp * freq_mask) 
        
        # 4. 恢复
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # 跨模态残差 (保持像素级权重)
        return out + F_rgb * illum + F_ir * (1 - illum)



# --- 1. 定义坐标注意力 (Coordinate Attention) ---
# 这是一个轻量级模块，能够帮助模型感知物体的"形状"和"位置"，
# 对于区分长宽比不同的车辆（如卡车 vs 面包车）非常有效。
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 两个方向的自适应池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 分别对 H 和 W 维度池化，保留空间结构信息
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # 在原特征上加权
        out = identity * a_w * a_h
        return out

# --- 2. 改进后的融合模块 V5 ---
class FrequencyCrossAttention_V5(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 像素级光照感知 (保持 V4 的优秀设计)
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 2. 改进：大感受野频域注意力
        # 将 kernel_size 增大到 7，帮助捕捉大物体(卡车/大巴)的低频结构特征
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # 3. 新增：坐标注意力增强
        # 在频域融合恢复后，再次进行空间-通道校准
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照感知 ---
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1)) # (B, 1, H, W)

        # --- B. 频域特征提取 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 预先在空间域利用光照权重进行一次粗融合，减少FFT计算量并聚焦重要区域
        # 这是一个 Trick: 先融合再变频，比变频后再融合更稳定
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # 转入频域
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # --- C. 频域增强 ---
        # 利用 IR 的纹理强度来生成 Mask (通常 IR 边缘在频域幅值上反应明显)
        # 这里我们也考虑 RGB 的贡献，取两者的最大幅值作为参考
        rgb_amp_ref = torch.abs(torch.fft.rfft2(Q_rgb, norm='backward'))
        ir_amp_ref = torch.abs(torch.fft.rfft2(K_ir, norm='backward'))
        max_amp = torch.max(rgb_amp_ref, ir_amp_ref)
        
        # 学习频率掩码
        freq_mask = self.spatial_gate(max_amp)
        
        # 增强幅值：保留原有幅值 + 掩码增强的高频部分
        enhanced_amp = fused_amp + (fused_amp * freq_mask)
        
        # --- D. 恢复与校准 ---
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # --- E. 坐标注意力校准 (关键改进) ---
        # 帮助模型区分长条形的卡车和短的货车
        out = self.ca(out)
        
        # 跨模态残差
        return out + F_rgb * illum + F_ir * (1 - illum)



# --- V6 ---
# --- 保持 CoordAtt 不变 (对长条形车有效) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- V6 核心组件：多尺度频域门控 ---
class MultiScaleFrequencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 分支1：小感受野，关注高频细节 (Van, Car)
        self.branch3x3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1)
        # 分支2：中感受野
        self.branch5x5 = nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2)
        # 分支3：大感受野，关注整体轮廓 (Truck, Bus)
        self.branch7x7 = nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3)
        
        # 融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(channels // 4 * 3, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch3x3(x)
        b2 = self.branch5x5(x)
        b3 = self.branch7x7(x)
        
        # 拼接不同尺度的特征
        concat = torch.cat([b1, b2, b3], dim=1)
        return self.fuse(concat)

class FrequencyCrossAttention_V6(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 像素级光照感知 (V4/V5 验证有效)
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 2. 改进：多尺度频域注意力 (Multi-Scale Frequency Gate)
        # 替代原本单一的 7x7 卷积
        self.ms_spatial_gate = MultiScaleFrequencyGate(hidden_channels)

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # 3. 坐标注意力 (V5 验证对 Freight_car 有效)
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照权重 ---
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # --- B. 特征投影 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 空间域预融合 (Efficient Fusion)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # --- C. 频域处理 ---
        # 1. FFT
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # 2. 生成多尺度频域掩码
        # 参考：同时利用 RGB 和 IR 的频域强度最大值来生成掩码
        rgb_amp_ref = torch.abs(torch.fft.rfft2(Q_rgb, norm='backward'))
        ir_amp_ref = torch.abs(torch.fft.rfft2(K_ir, norm='backward'))
        max_amp = torch.max(rgb_amp_ref, ir_amp_ref)
        
        # 使用多尺度模块处理幅值
        freq_mask = self.ms_spatial_gate(max_amp)
        
        # 3. 增强幅值
        enhanced_amp = fused_amp + (fused_amp * freq_mask)
        
        # --- D. 恢复 ---
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # --- E. 坐标校准 ---
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)

# ---   V7  ---
# --- 1. 保持 CoordAtt 不变 (对长条形车有效) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 改进：选择性多尺度频域门控 (Selective Kernel Frequency Gate) ---
class SKFrequencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        reduction = 4
        
        # 多尺度分支
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        # 注意力生成 (SK机制)
        # 将不同分支的信息聚合，生成选择权重
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 3, 1, bias=False) # 输出3个分支的权重
        )
        
        # 最终融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Split (多尺度特征)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        # 2. Fuse (聚合全局信息)
        feat_sum = feat3 + feat5 + feat7
        b, c, h, w = x.shape
        
        # 3. Select (生成权重)
        w_global = self.avg_pool(feat_sum)
        weights = self.fc(w_global)
        # Reshape to (B, 3, C, 1, 1) for softmax
        weights = weights.view(b, 3, c, 1, 1)
        weights = F.softmax(weights, dim=1)
        
        # 4. 加权求和
        # (B, C, H, W)
        feat_selected = (weights[:, 0, :, :, :] * feat3 + 
                         weights[:, 1, :, :, :] * feat5 + 
                         weights[:, 2, :, :, :] * feat7)
        
        return self.fuse_conv(feat_selected)

class FrequencyCrossAttention_V7(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 像素级光照感知
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 2. 改进：选择性多尺度频域门控
        self.sk_spatial_gate = SKFrequencyGate(hidden_channels)

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # 3. 坐标注意力
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照权重 ---
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1)) # (B, 1, H, W)

        # --- B. 特征投影 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 空间域预融合 (Base Fusion)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # --- C. 频域处理 ---
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # --- 关键改进点：更稳健的掩码生成输入 ---
        # 以前 V6 是取 max(rgb, ir)，容易引入噪声导致 Precision 下降
        # 现在 V7 采用 illum 加权的参考幅值，只增强我们"信任"的模态的频率特征
        # 注意：illum 是空间域的，这里我们近似认为它在低频段对频域也有指导意义，
        # 或者为了计算简便，我们直接用 fused_amp (它本身就是加权后的) 作为基础，
        # 再加上 IR 的高频特征（如果处于夜间）。
        
        # 策略 V7：直接使用 fused_amp 作为输入去学习 Mask。
        # 因为 fused_amp 已经包含了 illum 的加权信息，
        # SK-Gate 会自动判断这里的频率特征是否属于"有效特征"
        freq_mask = self.sk_spatial_gate(fused_amp)
        
        # 增强幅值
        enhanced_amp = fused_amp + (fused_amp * freq_mask)
        
        # --- D. 恢复 ---
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # --- E. 坐标校准 ---
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)


# --- v8
# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 改进：双路混合频域门控 (Dual-Path Hybrid Gate) ---
class DualPathFrequencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        reduction = 4
        
        # === Path A: Context Path (SK-Gate) ===
        # 负责稳健的特征提取，抑制噪声 (Precision)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 3, 1, bias=False)
        )
        
        # === Path B: Detail Path (Salience) ===
        # 负责捕获微弱的高频细节，捞回漏检目标 (Recall)
        # 不参与 SK 竞争，直接作为残差补充
        self.detail_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        # 最终融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- Path A: SK Context ---
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        feat_sum = feat3 + feat5 + feat7
        b, c, h, w = x.shape
        
        w_global = self.avg_pool(feat_sum)
        weights = self.fc(w_global)
        weights = weights.view(b, 3, c, 1, 1)
        weights = F.softmax(weights, dim=1)
        
        feat_context = (weights[:, 0, :, :, :] * feat3 + 
                        weights[:, 1, :, :, :] * feat5 + 
                        weights[:, 2, :, :, :] * feat7)
        
        # --- Path B: Detail Residual ---
        # 直接提取纹理，不被 Softmax 稀释
        feat_detail = self.detail_conv(x)
        
        # --- Fusion ---
        # 上下文 + 细节
        return self.fuse_conv(feat_context + feat_detail)

class FrequencyCrossAttention_V8(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 像素级光照感知
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 2. 改进：双路混合频域门控
        self.dual_spatial_gate = DualPathFrequencyGate(hidden_channels)

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # 3. 坐标注意力
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照权重 ---
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # --- B. 特征投影 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 空间域预融合 (Base Fusion)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # --- C. 频域处理 ---
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # --- 学习掩码 ---
        # 使用 V7 的稳健输入 (fused_amp)，但通过 V8 的双路门控处理
        # 既能看到大车的轮廓(SK)，也能看到小车的细节(Detail)
        freq_mask = self.dual_spatial_gate(fused_amp)
        
        # 增强幅值
        enhanced_amp = fused_amp + (fused_amp * freq_mask)
        
        # --- D. 恢复 ---
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # --- E. 坐标校准 ---
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)


# --- 1. CoordAtt (保持不变，作为空间校准) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 核心创新：可学习频域软阈值模块 ---
# (Learnable Frequency Soft-Thresholding)
class LearnableSoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 全局上下文建模：预测每个通道的阈值
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid() # 输出 0~1 之间的缩放系数
        )
        # 基础阈值参数 (可学习)
        self.threshold_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_amp):
        # x_amp: 频域幅值 (B, C, H, W)
        
        # 1. 计算动态阈值 (B, C, 1, 1)
        # 网络根据当前图像的内容，决定每个通道需要滤除多少噪声
        global_stat = self.gap(x_amp)
        dynamic_threshold = self.fc(global_stat) * self.threshold_scale
        
        # 2. 应用软阈值化公式
        # formula: y = sign(x) * max(|x| - threshold, 0)
        # 因为幅值总是正的，sign(x)为1，简化为: y = relu(x - threshold)
        
        # 为了保持梯度平滑，我们通常使用 softplus 或者直接 relu
        # 这里用 ReLU 实现截断
        x_amp_denoised = F.relu(x_amp - dynamic_threshold * x_amp)
        
        # 注意：这里我们做的是 "减去阈值比例"，
        # 这相当于一种自适应的 High-Pass Filter 的逆过程（去噪）
        
        return x_amp_denoised

class FrequencyCrossAttention_V10(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 1. 像素级光照感知 (保持稳健)
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 2. 改进：频域去噪与增强
        self.soft_threshold = LearnableSoftThresholding(hidden_channels)
        
        # 细节增强分支 (Detail Branch from V8/V9)
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # 3. 坐标注意力
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照权重 ---
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # --- B. 特征投影 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 空间域预融合
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # --- C. 频域软阈值去噪 (Thesis Core) ---
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # 1. 去噪：滤除背景杂波 (解决 Truck 破碎问题)
        amp_clean = self.soft_threshold(fused_amp)
        
        # 2. 增强：对保留下来的关键特征进行加权 (解决 Van 漏检问题)
        # 使用一个卷积层学习哪些频率是重要的细节
        detail_gain = self.detail_enhance(amp_clean)
        amp_final = amp_clean * (1 + detail_gain)
        
        # --- D. 恢复 ---
        combined_freq = torch.polar(amp_final, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # --- E. 坐标校准 ---
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)



# --- v11
# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 核心创新：频域感知软阈值 (Frequency-Aware Soft-Thresholding) ---
class FrequencyAwareSoftThreshold(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 1. 频率分离器 (通过不同卷积核模拟)
        # 低频保护通路 (7x7, 大感受野)
        self.low_freq_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        # 高频处理通路 (3x3, 小感受野)
        self.high_freq_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        # 2. 阈值预测器 (仅针对高频)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_thresh = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid() 
        )
        # 初始化阈值缩放因子，设小一点(0.1)，避免初始切除过多
        self.threshold_scale = nn.Parameter(torch.tensor(0.1))

        # 3. 融合权重
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x_amp):
        # x_amp: (B, C, H, W)
        
        # --- A. 分离频率 ---
        # 模拟低频部分 (主要能量/形状) -> 直接保留，保护 Truck
        amp_low = self.low_freq_conv(x_amp)
        
        # 模拟高频部分 (细节/噪声) -> 需要阈值化处理
        amp_high = self.high_freq_conv(x_amp)
        
        # --- B. 动态软阈值化 (仅对高频) ---
        # 计算阈值
        global_stat = self.gap(amp_high)
        # 阈值范围约束在 0~scale * max(amp) 之间
        thresh = self.fc_thresh(global_stat) * self.threshold_scale * x_amp.max()
        
        # 应用软阈值: sign(x) * max(|x| - thresh, 0)
        # 因为 amp >= 0，简化为 relu(amp - thresh)
        amp_high_clean = F.relu(amp_high - thresh)
        
        # --- C. 细节回补 (Residual Detail) ---
        # 这里的 trick 是：我们不仅去噪，还把去噪后的高频加权回去
        # 这样 Van 的轮廓（干净的高频）就被强化了
        
        # 融合：保护的低频 + 干净的高频
        amp_out = torch.cat([amp_low, amp_high_clean], dim=1)
        amp_out = self.fuse(amp_out)
        
        return amp_out

class FrequencyCrossAttention_V11(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 核心模块 V11
        self.fast_module = FrequencyAwareSoftThreshold(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # A. 光照权重
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # B. 频域预备
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # C. 频域感知软阈值处理 (V11 Core)
        # 既保护了低频(Truck)，又清洗了高频(Van/Freight)
        amp_enhanced = self.fast_module(fused_amp)
        
        # 这里的 amp_enhanced 可能经过 conv 后数值范围变了，
        # 建议加一个残差连接，保证训练初期不崩
        amp_final = fused_amp + amp_enhanced
        
        # D. 恢复
        combined_freq = torch.polar(amp_final, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)

# --- v12
import math
# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 核心创新：可学习频域位置掩码 (Learnable Frequency Position Mask) ---
class FrequencyPositionMask(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        # 频域的高度和宽度
        # rfft2 的输出宽度是 W//2 + 1
        self.h = height
        self.w = width // 2 + 1
        
        # 定义一个可学习的全局掩码参数 (1, 1, H, W_half)
        # 初始化为 1 (全通)，让网络慢慢学着去抑制高频噪声
        self.weight_mask = nn.Parameter(torch.ones(1, 1, self.h, self.w), requires_grad=True)
        
        # 也可以加一个通道级的缩放，增加灵活性
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x_amp):
        # x_amp: (B, C, H, W_half)
        
        # 1. 位置感知掩码
        # 网络会自动学习到: 左上角(低频)保持大值，右下角(高频)如果噪点多则变小
        mask = torch.sigmoid(self.weight_mask) # 约束在 0~1
        
        # 2. 通道自适应
        c_scale = self.channel_scale(x_amp)
        
        # 3. 最终加权
        # 这是"乘法"去噪，相当于 Soft Filter
        return x_amp * mask * c_scale

class FrequencyCrossAttention_V12(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 动态初始化：因为不知道 H, W 具体是多少 (YOLO是多尺度的)
        # 这里我们需要一个自适应的方案，或者固定一个最大尺寸进行插值
        # 为了兼容 YOLO 的多尺度输入，我们不能写死 Mask 的尺寸
        # 改进：使用 Parameter 生成器，或者直接在 forward 里动态插值 Mask
        self.freq_filter = LearnableFrequencyFilter(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # A. 光照权重
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # B. 频域预备
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # C. 可学习频域滤波 (V12)
        # 这里我们处理的是幅值
        amp_filtered = self.freq_filter(fused_amp)
        
        # 残差连接：保留原始信息，防止初始化阶段特征丢失
        # 这里的 amp_filtered 倾向于是一个"干净版"
        # 我们可以用 amp_filtered + fused_amp * 0.5 混合
        amp_final = fused_amp + amp_filtered
        
        # D. 恢复
        combined_freq = torch.polar(amp_final, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)

# --- 辅助类：自适应尺寸的可学习滤波器 ---
class LearnableFrequencyFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 定义一个足够大的基础 Mask (例如 256x256，覆盖大部分特征图尺寸)
        # YOLO P3/P4/P5 特征图大小不同，P3(80x80), P4(40x40), P5(20x20) @ 640imgsz
        # 我们定义一个通用的 Pattern Generator
        
        # 使用 MLP 生成 Mask 权重 (Implicit Neural Representation 思想)
        # 或者简单点：定义一个基础 Parameter，forward 时插值
        self.base_mask = nn.Parameter(torch.randn(1, 1, 128, 128) * 0.01, requires_grad=True) # 初始化接近0
        self.bias = nn.Parameter(torch.ones(1) * 0.5) # 加上偏置，初始接近 0.5~1.0
        
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x_amp):
        # x_amp: (B, C, H, W_freq)
        B, C, H, W_freq = x_amp.shape
        
        # 1. 动态插值 Mask 到当前特征图尺寸
        # base_mask (1, 1, 128, 128) -> (1, 1, H, W_freq)
        # 注意：频域只有一半宽度，我们假设 base_mask 覆盖全频域空间，这里裁剪或插值
        # 为了简单，直接插值到 (H, W_freq)
        
        mask = F.interpolate(self.base_mask, size=(H, W_freq), mode='bilinear', align_corners=False)
        mask = torch.sigmoid(mask + self.bias) 
        # 现在 mask 里的每个像素对应特定频率位置
        # 左上角对应低频
        
        # 2. 通道注意力
        c_gate = self.channel_gate(x_amp)
        
        # 3. 滤波
        # mask 决定保留哪些频率 (Spatial-wise in Frequency domain)
        # c_gate 决定哪些通道重要 (Channel-wise)
        return x_amp * mask * c_gate


# --- v13

# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 核心创新：基于显著性的门控模块 ---
class SaliencyGatedDetail(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 计算 Saliency 后用来生成门控权重
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 细节提取卷积
        self.detail_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        # 1. 计算粗略的显著性 (High-Pass like)
        # |x - smooth(x)| 捕捉边缘和纹理
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        saliency_map = torch.abs(x - smooth)
        
        # 2. 生成门控 (Where to inject detail?)
        # 平滑区域(Truck body) -> gate ~ 0
        # 边缘区域(Van contours) -> gate ~ 1
        gate = self.gate_conv(saliency_map)
        
        # 3. 提取细节
        detail = self.detail_conv(x)
        
        # 4. 门控注入
        return detail * gate

# --- 3. 改进的 V13 融合模块 ---
# 结合 V8 的稳健性(SK) + V13 的智能注入(Saliency)
class HybridSaliencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        reduction = 4
        
        # === Path A: Context Path (SK-Gate from V8) ===
        # 负责稳健的全局特征
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 3, 1, bias=False)
        )
        
        # === Path B: Saliency Detail Path (New) ===
        # 负责智能的细节注入
        self.saliency_detail = SaliencyGatedDetail(channels)
        
        # 最终融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- Path A: SK Context ---
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        feat_sum = feat3 + feat5 + feat7
        b, c, h, w = x.shape
        
        w_global = self.avg_pool(feat_sum)
        weights = self.fc(w_global)
        weights = weights.view(b, 3, c, 1, 1)
        weights = F.softmax(weights, dim=1)
        
        feat_context = (weights[:, 0, :, :, :] * feat3 + 
                        weights[:, 1, :, :, :] * feat5 + 
                        weights[:, 2, :, :, :] * feat7)
        
        # --- Path B: Saliency Detail ---
        # 只有在边缘处才有值
        feat_detail_gated = self.saliency_detail(x)
        
        # --- Fusion ---
        # Context + Masked Detail
        return self.fuse_conv(feat_context + feat_detail_gated)

class FrequencyCrossAttention_V13(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # V13 核心：基于显著性的混合门控
        self.hybrid_gate = HybridSaliencyGate(hidden_channels)

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # A. 光照权重
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # B. 频域预备 (这里我们还是用 FFT 来提取幅值，作为 Frequency Attention 的输入)
        # 虽然 Gate 内部有卷积，但输入是 Amplitude，所以依然是频域注意力
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # C. 学习掩码 (V13)
        # 输入幅值，利用 Saliency 机制决定哪里增强
        # 注意：这里我们是在 Frequency Domain 上做 "Conv" 
        # 虽然卷积在频域物理意义存疑，但在 V8 中证明了 "SK-Gate on Amp" 是有效的
        # V13 只是让 Detail 分支更聪明
        freq_mask = self.hybrid_gate(fused_amp)
        
        # 增强幅值
        enhanced_amp = fused_amp + (fused_amp * freq_mask)
        
        # D. 恢复
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)


# --- v14
# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 相位一致性计算模块 (保持不变) ---
class PhaseConsistencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.consistency_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Sigmoid()
        )

    def forward(self, amp_rgb, pha_rgb, amp_ir, pha_ir):
        # 1. 计算相位一致性 (0~1)
        phase_diff = torch.cos(pha_rgb - pha_ir)
        phase_consistency = torch.mean(phase_diff, dim=1, keepdim=True)
        phase_consistency = (phase_consistency + 1) / 2 
        consistency_map = self.consistency_conv(phase_consistency)
        
        # 2. 幅值融合决策
        amp_cat = torch.cat([amp_rgb, amp_ir], dim=1)
        base_weight = self.amp_fuse(amp_cat)
        final_weight = base_weight * (1 + consistency_map) # 一致性增强权重
        
        fused_amp = amp_rgb * final_weight + amp_ir * (1 - final_weight)
        
        return fused_amp, consistency_map

# --- 3. 修正后的 V14 主类 ---
class FrequencyCrossAttention_V14(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 光照感知
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 相位一致性门控
        self.phase_gate = PhaseConsistencyGate(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # A. 光照权重 (B, 1, H, W)
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # B. 频域提取
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 计算各自的频域特征
        freq_rgb = torch.fft.rfft2(Q_rgb, norm='backward')
        freq_ir = torch.fft.rfft2(K_ir, norm='backward')
        
        amp_rgb, pha_rgb = torch.abs(freq_rgb), torch.angle(freq_rgb)
        amp_ir, pha_ir = torch.abs(freq_ir), torch.angle(freq_ir)
        
        # C. 相位一致性协同融合 (计算融合后的幅值)
        fused_amp, consistency_map = self.phase_gate(amp_rgb, pha_rgb, amp_ir, pha_ir)
        
        # D. 复数重建 (修正点)
        # 错误代码: base_complex = freq_rgb * illum + freq_ir * (1 - illum)
        # 原因: illum是空间域(H,W)，freq_rgb是频域(H, W//2+1)，尺寸对不上。
        
        # 修正方案: 先在空间域融合，再转频域取相位
        # 这样既利用了光照权重，又避免了维度错误
        spatial_base = Q_rgb * illum + K_ir * (1 - illum)
        base_complex = torch.fft.rfft2(spatial_base, norm='backward')
        
        # 提取融合后的基准相位
        fused_pha = torch.angle(base_complex)
        
        # 结合：融合的幅值(来自相位一致性) + 融合的相位(来自光照加权)
        combined_freq = torch.polar(fused_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)


# --- v15
# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 改进的可学习频域滤波器 (V12 Pro) ---
class RobustFrequencyFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 基础掩码参数
        self.base_mask = nn.Parameter(torch.randn(1, 1, 128, 128) * 0.01, requires_grad=True)
        self.bias = nn.Parameter(torch.ones(1) * 0.5)
        
        # 通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # V8 的细节直通车 (Detail Path)
        # 用来“兜底”，防止 V12 的掩码把小车细节杀光了
        self.detail_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x_amp):
        B, C, H, W_freq = x_amp.shape
        
        # --- A. V12 位置掩码 ---
        # 插值
        mask = F.interpolate(self.base_mask, size=(H, W_freq), mode='bilinear', align_corners=False)
        mask = torch.sigmoid(mask + self.bias)
        
        # [关键修正]: 强制保护低频
        # 创建一个低频保护罩 (Low-Freq Shield)
        # 假设 H, W_freq 左上角是低频
        y = torch.linspace(-1, 1, H, device=x_amp.device).view(H, 1).repeat(1, W_freq)
        x = torch.linspace(-1, 1, W_freq, device=x_amp.device).view(1, W_freq).repeat(H, 1)
        dist = torch.sqrt(x*x + y*y) # 距离左上角的距离 (这里简化坐标系，实际FFT左边是低频)
        # 注意: rfft2 的布局是: W轴左边是0频率，H轴两端是0频率(如果fftshift了是中间)
        # PyTorch rfft2: H轴未shift (0和H-1是低频), W轴未shift (0是低频)
        
        # 我们用简单的硬编码：保护 H 轴的两头和 W 轴的左边
        # 这种硬编码比较难写通用，不如直接让 detail path 去补救
        
        # --- B. 滤波 ---
        c_gate = self.channel_gate(x_amp)
        amp_filtered = x_amp * mask * c_gate
        
        # --- C. V8 细节回补 ---
        # 这条路不经过 Mask，专门负责保留微弱的高频纹理
        # 注意：Detail conv 是在空间域定义的，这里输入是幅值(近似空间域特征)，可以混用
        # 或者为了严谨，我们在外部做 Detail 加法
        
        return amp_filtered

class FrequencyCrossAttention_V15(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # V15 滤波器
        self.freq_filter = RobustFrequencyFilter(hidden_channels)
        
        # V8 的 Detail 分支 (在 spatial domain 做)
        self.detail_branch = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.detail_gate = nn.Sequential(nn.Conv2d(hidden_channels, 1, 1), nn.Sigmoid())

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # A. 光照权重
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # B. 频域预备
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # --- C1. 细节分支 (Detail Path) ---
        # 趁着还没进频域，先提取空间细节
        # 这就是 V8 成功的关键：不把所有鸡蛋放在频域篮子里
        spatial_detail = self.detail_branch(fused_spatial_base)
        detail_weight = self.detail_gate(fused_spatial_base)
        
        # --- C2. 频域滤波 (V12 Path) ---
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # 应用 V12 掩码
        amp_filtered = self.freq_filter(fused_amp)
        
        # 恢复
        combined_freq = torch.polar(amp_filtered, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')
        
        # --- D. 融合 ---
        # 主干 (频域滤波后) + 细节 (空间域提取)
        # 这样既有 V12 的降噪 (High Precision)，又有 V8 的纹理 (High Recall)
        out_fused = out_spatial + spatial_detail * detail_weight

        out = self.output_conv(out_fused)
        out = self.norm(out)
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)


# --- 1. DWT 和 IDWT (保持不变) ---
class DWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, x):
        B, C, H, W = x.shape
        ll = F.conv2d(x, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        lh = F.conv2d(x, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        hl = F.conv2d(x, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        hh = F.conv2d(x, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return ll, lh, hl, hh

class IDWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        x_ll = F.conv_transpose2d(ll, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        x_lh = F.conv_transpose2d(lh, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        x_hl = F.conv_transpose2d(hl, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        x_hh = F.conv_transpose2d(hh, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return x_ll + x_lh + x_hl + x_hh

# --- 2. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

# --- 3. 修正后的 WaveletFusion (增加 Padding 逻辑) ---
class WaveletFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        
        # A. 低频处理 (LL)
        self.ll_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
        # B. 高频处理 (High-Freq)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_ir):
        # [Fix Start] 处理奇数尺寸问题
        B, C, H, W = x_rgb.shape
        pad_h = H % 2
        pad_w = W % 2
        
        # 如果高度或宽度是奇数，在右侧/下侧填充1个像素
        if pad_h or pad_w:
            x_rgb = F.pad(x_rgb, (0, pad_w, 0, pad_h), mode='replicate')
            x_ir = F.pad(x_ir, (0, pad_w, 0, pad_h), mode='replicate')
        # [Fix End]
        
        # 1. DWT 分解
        r_ll, r_lh, r_hl, r_hh = self.dwt(x_rgb)
        i_ll, i_lh, i_hl, i_hh = self.dwt(x_ir)
        
        # 2. 低频融合 (LL)
        ll_cat = torch.cat([r_ll, i_ll], dim=1)
        w_ll = self.ll_gate(ll_cat)
        f_ll = w_ll * r_ll + (1 - w_ll) * i_ll
        
        # 3. 高频融合 (High-Freq)
        def fuse_high_freq(h_rgb, h_ir):
            mag_rgb = torch.abs(h_rgb)
            mag_ir = torch.abs(h_ir)
            mask = torch.sigmoid(mag_rgb - mag_ir) 
            return mask * h_rgb + (1 - mask) * h_ir

        f_lh = fuse_high_freq(r_lh, i_lh)
        f_hl = fuse_high_freq(r_hl, i_hl)
        f_hh = fuse_high_freq(r_hh, i_hh)
        
        # 4. IDWT 重建
        out = self.idwt(f_ll, f_lh, f_hl, f_hh)
        
        # [Fix Start] 裁剪回原始尺寸
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        # [Fix End]
        
        return out

class FrequencyCrossAttention_V16(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        self.wavelet_fusion = WaveletFusion(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)
        
        self.illum_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 这里已经在 WaveletFusion 内部处理了 padding
        fused_feat = self.wavelet_fusion(Q_rgb, K_ir)
        
        out = self.output_conv(fused_feat)
        out = self.norm(out)
        out = self.ca(out)
        
        # 全局残差连接 (确保尺寸一致)
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))
        return out + F_rgb * illum + F_ir * (1 - illum)

# --- 1. DWT/IDWT (保持不变) ---
class DWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, x):
        B, C, H, W = x.shape
        # Pad if dimensions are odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        
        ll = F.conv2d(x, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        lh = F.conv2d(x, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        hl = F.conv2d(x, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        hh = F.conv2d(x, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return ll, lh, hl, hh, (H, W)

class IDWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, ll, lh, hl, hh, original_size):
        B, C, H, W = ll.shape
        x_ll = F.conv_transpose2d(ll, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        x_lh = F.conv_transpose2d(lh, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        x_hl = F.conv_transpose2d(hl, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        x_hh = F.conv_transpose2d(hh, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        
        out = x_ll + x_lh + x_hl + x_hh
        # Crop back to original size
        orig_H, orig_W = original_size
        return out[:, :, :orig_H, :orig_W]

# --- 2. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

# --- 3. V17: Wavelet + Detail Fusion ---
class WaveletDetailFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        
        # --- Path A: Wavelet (Structure Focus) ---
        # 1. LL Gate (保持 V16 的设计，保护 Truck)
        self.ll_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        # 高频部分简化处理，主要靠 Detail Path 补充
        
        # --- Path B: Spatial Detail (Texture Focus, from V8) ---
        # 专门提取小波可能忽略的微弱纹理 (Van)
        self.detail_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_ir, spatial_base):
        # x_rgb, x_ir: 降维后的特征
        # spatial_base: 预融合的空间特征 (illum * rgb + ...)
        
        # --- 1. Wavelet Path (处理结构) ---
        r_ll, r_lh, r_hl, r_hh, size = self.dwt(x_rgb)
        i_ll, i_lh, i_hl, i_hh, _    = self.dwt(x_ir)
        
        # LL Fusion (Smooth Weighting for Truck)
        ll_cat = torch.cat([r_ll, i_ll], dim=1)
        w_ll = self.ll_gate(ll_cat)
        f_ll = w_ll * r_ll + (1 - w_ll) * i_ll
        
        # High-Freq Fusion (Simple Max for Strong Edges)
        # 我们这里做一个简化，既然有 Detail Path 了，HF 可以简单点
        def fuse_hf(h1, h2):
            mask = (torch.abs(h1) > torch.abs(h2)).float()
            return mask * h1 + (1 - mask) * h2
            
        f_lh = fuse_hf(r_lh, i_lh)
        f_hl = fuse_hf(r_hl, i_hl)
        f_hh = fuse_hf(r_hh, i_hh)
        
        # Reconstruct
        out_wavelet = self.idwt(f_ll, f_lh, f_hl, f_hh, size)
        
        # --- 2. Detail Path (处理微弱纹理) ---
        # 这一步至关重要，它把 Van 的细节"加"回来
        detail = self.detail_conv(spatial_base)
        gate = self.detail_gate(spatial_base)
        out_detail = detail * gate
        
        # --- 3. Final Sum ---
        return out_wavelet + out_detail

class FrequencyCrossAttention_V17(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # V17 融合模块
        self.fusion = WaveletDetailFusion(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)
        
        self.illum_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        
        # 1. 降维
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 2. 准备 Base (用于 Detail 提取)
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # 3. V17 融合
        fused_feat = self.fusion(Q_rgb, K_ir, fused_spatial_base)
        
        # 4. 恢复
        out = self.output_conv(fused_feat)
        out = self.norm(out)
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)

# --- v18
# --- 1. DWT/IDWT (数学变换，保持不变) ---
class DWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        
        ll = F.conv2d(x, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        lh = F.conv2d(x, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        hl = F.conv2d(x, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        hh = F.conv2d(x, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return ll, lh, hl, hh, (H, W)

class IDWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, ll, lh, hl, hh, original_size):
        B, C, H, W = ll.shape
        x_ll = F.conv_transpose2d(ll, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        x_lh = F.conv_transpose2d(lh, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        x_hl = F.conv_transpose2d(hl, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        x_hh = F.conv_transpose2d(hh, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        
        out = x_ll + x_lh + x_hl + x_hh
        orig_H, orig_W = original_size
        return out[:, :, :orig_H, :orig_W]

# --- 2. 基础组件 ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

# 轻量级 SE 模块 (用于 Detail Branch)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- 3. V18: Soft-Wavelet + Attentive Detail ---
class SoftWaveletDetailFusionv18(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        
        # A. Low-Freq Fusion (Truck Structure)
        self.ll_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
        # B. Attentive Detail Path (Van Texture)
        # 增加 SE 模块，让 Detail 分支更具选择性
        self.detail_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            # nn.GroupNorm(4, channels),
            nn.ReLU(),
            SEBlock(channels)  # [Improvement] Smart selection of details
        )
        self.detail_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        
        # C. Balance Weight
        # 学习一个标量系数，决定 Detail 加多少
        # self.detail_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.detail_scale = nn.Parameter(torch.zeros(1))

        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x_rgb, x_ir, spatial_base):
        # 1. Wavelet Decomposition
        r_ll, r_lh, r_hl, r_hh, size = self.dwt(x_rgb)
        i_ll, i_lh, i_hl, i_hh, _    = self.dwt(x_ir)
        
        # 2. LL Fusion (Structure)
        ll_cat = torch.cat([r_ll, i_ll], dim=1)
        w_ll = self.ll_gate(ll_cat)
        f_ll = w_ll * r_ll + (1 - w_ll) * i_ll
        
        # 3. High-Freq Fusion (Soft Gating)
        # [Fix] 使用 Soft Sigmoid 替代 Hard Max，梯度更顺滑
        def fuse_soft(h1, h2):
            # 比较两者幅值，生成软掩码
            mag1 = torch.abs(h1)
            mag2 = torch.abs(h2)
            # 乘以 10 是为了增加区分度，模拟 Soft-argmax
            mask = torch.sigmoid((mag1 - mag2) * self.temp) 
            return mask * h1 + (1 - mask) * h2

        f_lh = fuse_soft(r_lh, i_lh)
        f_hl = fuse_soft(r_hl, i_hl)
        f_hh = fuse_soft(r_hh, i_hh)
        
        # 4. Wavelet Reconstruction
        out_wavelet = self.idwt(f_ll, f_lh, f_hl, f_hh, size)
        
        # 5. Detail Path
        detail = self.detail_branch(spatial_base)
        gate = self.detail_gate(spatial_base)
        out_detail = detail * gate
        
        # 6. Weighted Sum
        return out_wavelet + self.detail_scale * out_detail

class FrequencyCrossAttention_V18(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # V18 Fusion
        self.fusion = SoftWaveletDetailFusion(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)
        
        self.illum_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # V18 Fusion
        fused_feat = self.fusion(Q_rgb, K_ir, fused_spatial_base)
        
        out = self.output_conv(fused_feat)
        out = self.norm(out)
        out = self.ca(out)

        # out_rgb = F_rgb + out * illum       # RGB 吸收融合后的特征（偏向 RGB 的权重）
        # out_ir  = F_ir  + out * (1 - illum) # IR  吸收融合后的特征（偏向 IR 的权重）
        
        return out + F_rgb * illum + F_ir * (1 - illum)
        # return out_rgb,out_ir


class base_ADD(nn.Module):
    #  Add two tensors
    
    def __init__(self, arg):
        super().__init__()
        # 128 256 512
        self.arg = arg
  
    def forward(self, x):
        return torch.add(x[0], x[1])


class CAFusionv1(nn.Module):
    """
    Cross-Modal Coordinate Attention Fusion (CMCA)
    用于 Backbone 阶段的空间对齐与特征校准。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 1. 坐标池化 (保持 CoordAtt 的核心优势：保留位置信息)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 2. 共享特征变换 (处理拼接后的特征)
        # 输入是 2 * channels (RGB+IR)，中间压缩
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels * 2, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # 3. 分离注意力生成 (分别生成 RGB 和 IR 的权重)
        # 输出通道为 2 * channels (分别对应 RGB 和 IR)
        self.conv_h = nn.Conv2d(mip, channels * 2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x_rgb, x_ir):
        # x 应该是一个列表 [rgb, ir]
        # x_rgb, x_ir = x[0], x[1]
        B, C, H, W = x_rgb.shape

        # --- A. 联合特征编码 ---
        # 在空间上拼接，让网络同时看到两个模态的信息
        x_cat = torch.cat([x_rgb, x_ir], dim=1)  # (B, 2C, H, W)

        # --- B. 坐标注意力提取 ---
        # 1. X 方向和 Y 方向池化
        x_h = self.pool_h(x_cat)  # (B, 2C, H, 1)
        x_w = self.pool_w(x_cat).permute(0, 1, 3, 2)  # (B, 2C, 1, W)

        # 2. 拼接并降维 (捕捉跨模态的长距离依赖)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 

        # 3. 拆分回 H 和 W
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # --- C. 交叉注意力生成 ---
        # 生成 H 方向的权重 (B, 2C, H, 1)
        a_h = self.conv_h(x_h).sigmoid()
        # 生成 W 方向的权重 (B, 2C, 1, W)
        a_w = self.conv_w(x_w).sigmoid()

        # --- D. 权重拆分与应用 ---
        # 将 2C 的权重拆分为 RGB权重 和 IR权重
        a_h_rgb, a_h_ir = torch.split(a_h, C, dim=1)
        a_w_rgb, a_w_ir = torch.split(a_w, C, dim=1)

        # 应用权重 (Cross-Calibration)
        # 网络会自动学习：如果 RGB 是黑的(无信息)，RGB权重会变小，利用 IR 的信息增强
        out_rgb = x_rgb * a_w_rgb * a_h_rgb
        out_ir = x_ir * a_w_ir * a_h_ir

        # 返回列表，保持双流结构继续向下传递
        return out_rgb, out_ir



class CAFusionv2(nn.Module):
    """
    Frequency-Aware Interactive Enhancement (FAIE)
    学术名称: 频率感知交互增强模块
    作用: 在Backbone阶段,将RGB的高频(纹理)借给IR,将IR的低频(结构)借给RGB。
    """
    """
    Frequency-Aware Interactive Enhancement (FAIE) - Robust Version
    [Fix]: 修复了奇数输入尺寸导致的维度不匹配问题 (e.g., 21 vs 20)
    """

    """
    Spatially Adaptive FAIE (SA-FAIE)
    [Upgrade]: 增加了空间门控 (Spatial Gate)，防止背景噪声过度注入。
    解决了在大规模测试集上 Van 类别的纹理过拟合问题。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # 1. 频率提取器 (保持不变)
        self.lpf = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 2. 特征变换与门控 (升级部分)
        # RGB增强分支 (接收IR低频): 1x1 Conv 调整特征 + 3x3 Conv 生成空间掩码
        self.conv_ir_low = nn.Conv2d(channels, channels, 1)
        self.gate_ir = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # IR增强分支 (接收RGB高频): 同上
        self.conv_rgb_high = nn.Conv2d(channels, channels, 1)
        self.gate_rgb = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 3. 可学习系数 (保持 Zero-Init)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2=None):
        if x2 is not None:
            x_rgb, x_ir = x1, x2
        else:
            x_rgb, x_ir = x1[0], x1[1]

        H, W = x_rgb.shape[2:]

        # --- A. 频率分离 ---
        # 1. 提取 IR 低频 (Structure)
        ir_low = F.interpolate(self.lpf(x_ir), size=(H, W), mode='nearest')
        
        # 2. 提取 RGB 高频 (Texture)
        rgb_low = F.interpolate(self.lpf(x_rgb), size=(H, W), mode='nearest')
        rgb_high = x_rgb - rgb_low

        # --- B. 空间自适应交叉注入 ---
        
        # 1. 增强 RGB: 注入 IR 的结构
        # 先计算注入特征
        feat_to_inject_rgb = self.conv_ir_low(ir_low)
        # 再计算空间掩码 (Attention Map)
        mask_rgb = self.gate_ir(feat_to_inject_rgb)
        # 门控过滤: 只在重要区域注入结构
        enhance_rgb = feat_to_inject_rgb * mask_rgb
        out_rgb = x_rgb + self.gamma_rgb * enhance_rgb

        # 2. 增强 IR: 注入 RGB 的纹理
        # 先计算注入特征
        feat_to_inject_ir = self.conv_rgb_high(rgb_high)
        # 再计算空间掩码
        mask_ir = self.gate_rgb(feat_to_inject_ir)
        # 门控过滤: 过滤掉背景的杂乱纹理噪声!
        enhance_ir = feat_to_inject_ir * mask_ir
        out_ir = x_ir + self.gamma_ir * enhance_ir

        return out_rgb, out_ir

# =========================================================================
# 改进一：动态滤波器 (Dynamic LPF) - 替代 AvgPool
# =========================================================================
class ContentAwareLPF(nn.Module):
    """
    内容感知低通滤波器 (Content-Aware Low Pass Filter)
    作用: 使用可学习的 Depthwise Conv 替代固定的 AvgPool，
          自适应地提取低频结构，避免小目标在平均池化中消失。
    """
    def __init__(self, channels):
        super().__init__()
        # 使用 Depthwise Convolution，groups=channels，参数量极小
        # kernel=3, stride=2, padding=1 实现 2倍下采样
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, 
                                 groups=channels, bias=False)
        
        # 初始化技巧：初始化为类似高斯模糊的权重，保证初始状态稳定
        nn.init.constant_(self.dw_conv.weight, 1.0 / 9.0)

    def forward(self, x):
        return self.dw_conv(x)

class CAFusionv3(nn.Module):
    """
    [Upgrade]: Frequency-Aware Interactive Enhancement (FAIE) - Dynamic Version
    改进点: 
    1. LPF 升级为 ContentAwareLPF (动态滤波器)
    2. 上采样升级为 Bilinear (消除高频差分时的网格噪声)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.lpf = nn.AvgPool2d(kernel_size=2, stride=2)

        # 1. 频率提取器 (升级为动态可学习)
        # self.lpf = ContentAwareLPF(channels)
        
        # 2. 特征变换与门控
        # RGB增强分支 (接收IR低频)
        self.conv_ir_low = nn.Conv2d(channels, channels, 1)
        self.gate_ir = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # IR增强分支 (接收RGB高频)
        self.conv_rgb_high = nn.Conv2d(channels, channels, 1)
        self.gate_rgb = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 3. 可学习系数
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

        # [新增] 用于可视化的容器，设为 None
        self.vis_mask_rgb = None
        self.vis_mask_ir = None

    def forward(self, x1, x2=None):
        # 支持 YOLO 列表输入或直接双张量输入
        if x2 is not None:
            x_rgb, x_ir = x1, x2
        else:
            x_rgb, x_ir = x1[0], x1[1]

        H, W = x_rgb.shape[2:]

        # --- A. 频率分离 (动态版) ---
        # 1. 提取 IR 低频 (Structure)
        # 使用 bilinear 插值，避免 nearest 带来的马赛克锯齿，使差分得到的高频更纯净
        ir_low_feat = self.lpf(x_ir)
        ir_low = F.interpolate(ir_low_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 2. 提取 RGB 高频 (Texture)
        rgb_low_feat = self.lpf(x_rgb)
        rgb_low = F.interpolate(rgb_low_feat, size=(H, W), mode='bilinear', align_corners=False)
        rgb_high = x_rgb - rgb_low

        # --- B. 空间自适应交叉注入 ---
        
        # 1. 增强 RGB: 注入 IR 的结构
        feat_to_inject_rgb = self.conv_ir_low(ir_low)
        mask_rgb = self.gate_ir(feat_to_inject_rgb)

        # [新增] 关键步骤：detach并存入self，避免梯度问题，节省显存
        if not self.training: # 只在推理/验证模式下保存，训练时不管
            self.vis_mask_rgb = mask_rgb.detach()

        enhance_rgb = feat_to_inject_rgb * mask_rgb
        out_rgb = x_rgb + self.gamma_rgb * enhance_rgb

        # 2. 增强 IR: 注入 RGB 的纹理
        feat_to_inject_ir = self.conv_rgb_high(rgb_high)
        mask_ir = self.gate_rgb(feat_to_inject_ir)

        # [新增] 关键步骤
        if not self.training:
            self.vis_mask_ir = mask_ir.detach()

        enhance_ir = feat_to_inject_ir * mask_ir
        out_ir = x_ir + self.gamma_ir * enhance_ir

        return out_rgb, out_ir

# =========================================================================
# 改进二：方向感知性 (Directional Awareness) - 针对 OBB 任务
# =========================================================================

# 必要的组件 (DWT/IDWT, SEBlock 等) 保持你原有的代码不变，这里省略以节省篇幅
# 请保留你原代码中的 DWT_2D, IDWT_2D, SEBlock, CoordAtt 类定义

class DirectionalHighFreqGate(nn.Module):
    """
    方向感知门控 (Directional High-Frequency Gate)
    作用: 使用大长径比卷积 (Strip Conv) 感知 OBB 目标的旋转特征
    """
    def __init__(self, channels):
        super().__init__()
        # 水平方向感知 (1x5)
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        # 垂直方向感知 (5x1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        # 温度系数
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, h_rgb, h_ir):
        # 1. 分别提取两个模态的方向特征强度
        feat_rgb = self.conv_h(h_rgb) + self.conv_v(h_rgb)
        feat_ir  = self.conv_h(h_ir)  + self.conv_v(h_ir)
        
        # 2. 计算幅值差异
        mag_rgb = torch.abs(feat_rgb)
        mag_ir  = torch.abs(feat_ir)
        
        # 3. 生成方向感知的软掩码
        mask = torch.sigmoid((mag_rgb - mag_ir) * self.temp)
        
        # 4. 融合
        return mask * h_rgb + (1 - mask) * h_ir

class SoftWaveletDetailFusion(nn.Module):
    """
    [Upgrade]: Soft-Wavelet Fusion with Directional Awareness
    改进点: 
    1. 高频融合引入 DirectionalHighFreqGate, 针对 OBB 旋转边缘优化
    2. Detail Branch 保持原有 SEBlock 设计
    """
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        
        # A. Low-Freq Fusion (保持不变)
        self.ll_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
        # B. High-Freq Fusion (升级为方向感知门控)
        # 针对 LH, HL, HH 三个子带共享同一个门控逻辑，节省参数
        self.directional_gate = DirectionalHighFreqGate(channels)
        
        # C. Attentive Detail Path (保持不变)
        self.detail_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SEBlock(channels)
        )
        self.detail_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        
        self.detail_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x_rgb, x_ir, spatial_base):
        # 1. Wavelet Decomposition
        r_ll, r_lh, r_hl, r_hh, size = self.dwt(x_rgb)
        i_ll, i_lh, i_hl, i_hh, _    = self.dwt(x_ir)
        
        # 2. LL Fusion (Structure - 简单加权)
        ll_cat = torch.cat([r_ll, i_ll], dim=1)
        w_ll = self.ll_gate(ll_cat)
        f_ll = w_ll * r_ll + (1 - w_ll) * i_ll
        
        # 3. High-Freq Fusion (Directional Aware)
        # 使用方向感知门控处理高频子带
        f_lh = self.directional_gate(r_lh, i_lh)
        f_hl = self.directional_gate(r_hl, i_hl)
        f_hh = self.directional_gate(r_hh, i_hh)
        
        # 4. Wavelet Reconstruction
        out_wavelet = self.idwt(f_ll, f_lh, f_hl, f_hh, size)
        
        # 5. Detail Path
        detail = self.detail_branch(spatial_base)
        gate = self.detail_gate(spatial_base)
        out_detail = detail * gate
        
        # 6. Weighted Sum
        return out_wavelet + self.detail_scale * out_detail

# FrequencyCrossAttention_V18 类保持原样，
# 它会自动调用更新后的 SoftWaveletDetailFusion
class FrequencyCrossAttention_V19(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        # 这里会自动使用上面定义的新版 SoftWaveletDetailFusion
        self.fusion = SoftWaveletDetailFusion(hidden_channels)
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ca = CoordAtt(channels, channels)
        
        self.illum_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        fused_feat = self.fusion(Q_rgb, K_ir, fused_spatial_base)
        
        out = self.output_conv(fused_feat)
        out = self.norm(out)
        out = self.ca(out)

        return out + F_rgb * illum + F_ir * (1 - illum)


# # 别名
# class ADD(FrequencyCrossAttention_V19):
#     def __init__(self, channels, reduction_ratio=4):
#         super().__init__(channels, reduction_ratio)



# --- 1. CoordAtt (保持不变) ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# --- 2. 改进：双路混合频域门控 (带监控功能) ---

class DualPathFrequencyGate(nn.Module):
    def __init__(self, channels, debug=False):
        super().__init__()
        self.channels = channels
        self.debug = debug
        
        reduction = 4
        
        # === Path A: Context Path (SK-Gate) ===
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels * 3, 1, bias=False)
        )
        
        # === Path B: Detail Path (Salience) ===
        self.detail_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        # [保持] 零初始化策略 (防止 Ratio 爆炸)
        nn.init.constant_(self.detail_conv.weight, 0)
        nn.init.constant_(self.detail_conv.bias, 0)
        
        # 最终融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- Path A: SK Context ---
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        feat_sum = feat3 + feat5 + feat7
        b, c, h, w = x.shape
        
        w_global = self.avg_pool(feat_sum)
        weights = self.fc(w_global)
        weights = weights.view(b, 3, c, 1, 1)
        weights = F.softmax(weights, dim=1)
        
        feat_context = (weights[:, 0, :, :, :] * feat3 + 
                        weights[:, 1, :, :, :] * feat5 + 
                        weights[:, 2, :, :, :] * feat7)
        
        # --- Path B: Detail Residual ---
        feat_detail = self.detail_conv(x)
        
        # --- 监控逻辑 (保持不变) ---
        if self.debug and self.training:
            if torch.rand(1).item() < 0.01: 
                with torch.no_grad():
                    ctx_mean = feat_context.abs().mean().item()
                    det_mean = feat_detail.abs().mean().item()
                    ratio = det_mean / (ctx_mean + 1e-6)
                    sk_dist = weights.mean(dim=(0, 2, 3, 4)).tolist()
                    
                    print(f"\n[DualPath Monitor V8.1] Context: {ctx_mean:.4f} | Detail: {det_mean:.4f} | Ratio: {ratio:.2%}")
                    print(f"[SK Weights] 3x3: {sk_dist[0]:.2f}, 5x5: {sk_dist[1]:.2f}, 7x7: {sk_dist[2]:.2f}")
        
        # --- Fusion ---
        return self.fuse_conv(feat_context + feat_detail)

class FrequencyCrossAttention_V8_2(nn.Module):
    def __init__(self, channels, reduction_ratio=4, debug=True): # <--- [修改] 接口透传
        super().__init__()
        self.channels = channels
        hidden_channels = channels // reduction_ratio
        
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        
        self.illum_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        # 传递 debug 参数
        self.dual_spatial_gate = DualPathFrequencyGate(hidden_channels, debug=debug)

        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        self.ca = CoordAtt(channels, channels)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape

        # --- A. 光照权重 ---
        illum = self.illum_gate(torch.cat([F_rgb, F_ir], dim=1))

        # --- B. 特征投影 ---
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        fused_spatial_base = illum * Q_rgb + (1 - illum) * K_ir
        
        # --- C. 频域处理 ---
        fused_freq = torch.fft.rfft2(fused_spatial_base, norm='backward')
        fused_amp = torch.abs(fused_freq)
        fused_pha = torch.angle(fused_freq)
        
        # 学习掩码 (监控点在这里触发)
        freq_mask = self.dual_spatial_gate(fused_amp)
        
        enhanced_amp = fused_amp + (fused_amp * freq_mask)
        
        # --- D. 恢复 ---
        combined_freq = torch.polar(enhanced_amp, fused_pha)
        out_spatial = torch.fft.irfft2(combined_freq, s=(H, W), norm='backward')

        out = self.output_conv(out_spatial)
        out = self.norm(out)
        
        # --- E. 坐标校准 ---
        out = self.ca(out)
        
        return out + F_rgb * illum + F_ir * (1 - illum)

# 别名
# class ADD(FrequencyCrossAttention_V8_2):
#     def __init__(self, channels, reduction_ratio=4):
#         super().__init__(channels, reduction_ratio)


# =========================================================================
# v20

class DoGFilter(nn.Module):
    """
    高斯差分滤波器 (Difference of Gaussians) - 替代 ContentAwareLPF
    作用：通过不同尺度的平滑图像相减，精确提取高频边缘和纹理。 [cite: 174, 176]
    """
    def __init__(self, channels):
        super().__init__()
        # 模拟两个尺度的平滑 (sigma1 < sigma2)
        self.conv_s1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv_s2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        nn.init.constant_(self.conv_s1.weight, 1.0 / 9.0)
        nn.init.constant_(self.conv_s2.weight, 1.0 / 25.0)

    def forward(self, x):
        # DoG(x) = Gaussian_s1(x) - Gaussian_s2(x)
        return self.conv_s1(x) - self.conv_s2(x)

class SACFusion_v20(nn.Module):
    """
    [Upgrade]: Structured Awareness Cross-Fusion (SACFusion)
    改进点：
    1. 使用 DoGFilter 提取边缘细节 [cite: 9]
    2. 引入 SMMM 风格的多尺度空间显著性掩码 
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.dog = DoGFilter(channels)
        
        # 显著性建模：多尺度感受野 (3x3 & 5x5) [cite: 200, 203]
        self.saliency_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1, groups=channels // reduction),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        self.saliency_ir = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.Conv2d(channels // reduction, channels // reduction, 5, padding=2, groups=channels // reduction),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2=None):
        x_rgb, x_ir = (x1, x2) if x2 is not None else (x1[0], x1[1])
        
        # --- A. 结构化特征提取 ---
        # 提取 RGB 纹理细节和 IR 结构细节
        rgb_detail = self.dog(x_rgb)
        ir_detail = self.dog(x_ir)

        # --- B. 空间显著性掩码注入 ---
        # 1. 增强 RGB: 注入 IR 的显著性结构
        mask_ir = self.saliency_ir(x_ir)
        out_rgb = x_rgb + self.gamma_rgb * (ir_detail * mask_ir)

        # 2. 增强 IR: 注入 RGB 的显著性纹理
        mask_rgb = self.saliency_rgb(x_rgb)
        out_ir = x_ir + self.gamma_ir * (rgb_detail * mask_rgb)

        return out_rgb, out_ir




class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
    

class ACFA_DirectionalGate(nn.Module):
    """
    自适应交叉融合方向门控 (Adaptive Cross-Fusion Directional Gate)
    改进：支持动态尺寸适配，无需手动为每层设置固定的 hw
    """
    def __init__(self, channels, base_hw=32):
        super().__init__()
        self.c = channels
        c4 = channels // 4
        
        # 1. 定义可学习的引导张量 (Learnable Guidance)
        # 使用 base_hw 作为初始化参考尺寸
        self.param_hw = nn.Parameter(torch.randn(1, c4, base_hw, base_hw) * 0.02) # 平面
        self.param_h = nn.Parameter(torch.randn(1, c4, base_hw, 1) * 0.02)      # 水平
        self.param_w = nn.Parameter(torch.randn(1, c4, 1, base_hw) * 0.02)      # 垂直
        
        # 2. 深度卷积处理逻辑
        self.dw_hw = nn.Conv2d(c4, c4, 3, padding=1, groups=c4)
        self.dw_h = nn.Conv2d(c4, c4, (3, 1), padding=(1, 0), groups=c4)
        self.dw_w = nn.Conv2d(c4, c4, (1, 3), padding=(0, 1), groups=c4)
        
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x_rgb, x_ir):
        B, C, H, W = x_rgb.shape
        c4 = C // 4
        
        # 计算两个模态的差异作为引导基础
        diff = torch.abs(x_rgb - x_ir)
        x_split = torch.split(diff, [c4, c4, c4, C - 3*c4], dim=1)
        
        # --- 核心改进：动态对齐引导张量 ---
        # 无论输入 H, W 是多少，都将参数插值到当前尺寸
        g_hw = self.dw_hw(F.interpolate(self.param_hw, size=(H, W), mode='bilinear', align_corners=False))
        g_h  = self.dw_h(F.interpolate(self.param_h, size=(H, W), mode='bilinear', align_corners=False))
        g_w  = self.dw_w(F.interpolate(self.param_w, size=(H, W), mode='bilinear', align_corners=False))
        
        # 应用方向性加权
        feat_hw = x_split[0] * torch.sigmoid(g_hw)
        feat_h  = x_split[1] * torch.sigmoid(g_h)
        feat_w  = x_split[2] * torch.sigmoid(g_w)
        feat_gen = x_split[3] # 通用上下文分支
        
        # 合并并生成最终门控掩码
        mask = self.final_conv(torch.cat([feat_hw, feat_h, feat_w, feat_gen], dim=1))
        mask = torch.sigmoid(mask)
        
        return mask * x_rgb + (1 - mask) * x_ir

class FrequencyCrossAttention_V20(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        # 减少通道以降低计算量
        hidden_channels = channels // reduction_ratio
        self.proj_rgb = nn.Conv2d(channels, hidden_channels, 1)
        self.proj_ir = nn.Conv2d(channels, hidden_channels, 1)
        
        # 传入 base_hw=32 即可，模块内部会自动处理 P3-P5 的尺寸变化
        self.directional_fusion = ACFA_DirectionalGate(hidden_channels, base_hw=32)
        
        # SMMM 思想：多尺度空间掩码（使用扩张卷积增强局部连续性）
        self.saliency_mask = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2, groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv2d(hidden_channels, channels, 1)
        self.ca = CoordAtt(channels, channels) # 保持你原有的坐标注意力

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        
        # 投影到低维空间
        Q_rgb = self.proj_rgb(F_rgb)
        K_ir = self.proj_ir(F_ir)
        
        # 1. 结构化方向感知融合 (核心改进)
        fused_feat = self.directional_fusion(Q_rgb, K_ir)
        
        # 2. 空间显著性过滤 (SMMM)
        mask = self.saliency_mask(fused_feat)
        fused_feat = fused_feat * mask
        
        # 3. 输出重构与全局增强
        out = self.output_conv(fused_feat)
        return self.ca(out + F_rgb + F_ir)





class SACFusion(nn.Module):
    """
    [Backbone 专用]: 基于物理先验的边缘增强融合模块
    创新点：使用 DoG 算子进行局部频域分析，替代效果不佳的 LPF
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 1. DoG 频率滤波器：模拟低通滤波相减得到带通效果
        self.s1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.s2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        nn.init.constant_(self.s1.weight, 1.0 / 9.0)
        nn.init.constant_(self.s2.weight, 1.0 / 25.0)

        # 2. 空间显著性掩码 (借鉴 SMMM 思想)
        self.gate_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        self.gate_ir = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2=None):
        x_rgb, x_ir = (x1, x2) if x2 is not None else (x1[0], x1[1])
        
        # 提取高频边缘细节 (DoG)
        rgb_detail = self.s1(x_rgb) - self.s2(x_rgb)
        ir_detail = self.s1(x_ir) - self.s2(x_ir)

        # 显著性权重注入
        out_rgb = x_rgb + self.gamma_rgb * (ir_detail * self.gate_ir(x_ir))
        out_ir = x_ir + self.gamma_ir * (rgb_detail * self.gate_rgb(x_rgb))

        return out_rgb, out_ir
    
class ACFA_MCM_Fusion(nn.Module):
    """
    [Head 专用]: 结构化 3D 语义魔方融合模块
    创新点：整合方向感知引导 (ACFA) 与 3D 体素关联 (MCM)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.hidden = channels // reduction
        
        # 1. ACFA 方向引导分支 (适配尺寸动态插值)
        self.p_hw = nn.Parameter(torch.randn(1, self.hidden, 32, 32) * 0.02)
        self.p_h  = nn.Parameter(torch.randn(1, self.hidden, 32, 1) * 0.02)
        self.dw_h = nn.Conv2d(self.hidden, self.hidden, (3, 1), padding=(1, 0), groups=self.hidden)

        # 2. MCM 3D 关联分支 (X, Y, Z 三轴关联)
        # 简化版空间关联 (轴向解耦)
        self.proj_xy = nn.Conv2d(channels, self.hidden, 1)
        # 通道关联 (Z轴)
        self.proj_z = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.hidden, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden, channels, 1),
            nn.Sigmoid()
        )
        
        self.output = nn.Conv2d(self.hidden, channels, 1)

    def forward(self, x):
        F_rgb, F_ir = x[0], x[1]
        B, C, H, W = F_rgb.shape
        
        # --- A. 跨模态 3D 关联 (MCM 核心逻辑) ---
        # 1. 空间平面关联 (XY)
        feat_cat = self.proj_xy(F_rgb + F_ir)
        x_h = torch.mean(feat_cat, dim=2, keepdim=True) # B, C', 1, W
        x_v = torch.mean(feat_cat, dim=3, keepdim=True) # B, C', H, 1
        
        # 2. 方向引导注入 (ACFA 核心逻辑)
        # 动态对齐水平引导张量到当前尺寸
        g_h = self.dw_h(F.interpolate(self.p_h, size=(H, 1), mode='bilinear'))
        feat_spatial = (x_h * torch.sigmoid(g_h)) * x_v # 广播机制形成 3D 体素底座

        # 3. 通道关联注入 (Z轴)
        # 基于空间底座计算跨通道权重
        z_weight = self.proj_z(F_rgb + F_ir) 
        
        # --- B. 最终重构 ---
        fused = self.output(feat_spatial) * z_weight
        
        # 残差连接：结合原始特征与 3D 关联增强特征
        return fused + F_rgb + F_ir

class SFCA_Fusion_V2(nn.Module):
    """
    SFCA-Fusion V2: 引入动态相位重塑的全域融合模块
    核心创新：
    1. [振幅域]: 动态细节提纯 (FCA)
    2. [相位域]: 跨模态结构自适应交互 (Phase-Interaction) -> 解决结构失真
    3. [空间域]: 3D 轴向坐标关联 (ACFA-MCM) -> 解决语义对齐
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.hidden = channels // reduction
        
        # --- 1. 光照与置信度感知 (决定相位和振幅的权重) ---
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden, 1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, 2, 3, padding=1), # 输出两个 Mask：一个给振幅，一个给相位
            nn.Sigmoid()
        )

        # --- 2. 频域处理组件 ---
        self.amp_refine = nn.Sequential(
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1, groups=self.hidden),
            nn.Sigmoid()
        )
        
        # --- 3. 空间坐标关联 (ACFA-MCM) ---
        self.proj_rgb = nn.Conv2d(channels, self.hidden, 1)
        self.proj_ir = nn.Conv2d(channels, self.hidden, 1)
        
        self.p_h = nn.Parameter(torch.randn(1, self.hidden, 32, 1) * 0.02)
        self.dw_h = nn.Conv2d(self.hidden, self.hidden, (3, 1), padding=(1, 0), groups=self.hidden)

        self.output = nn.Sequential(
            nn.Conv2d(self.hidden, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        f_rgb, f_ir = x[0], x[1]
        b, c, h, w = f_rgb.shape

        # === Step A: 特征降维与双模态交互门控 ===
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        
        # 得到动态权重：masks[:, 0] 为振幅权重, masks[:, 1] 为相位置信度
        masks = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        alpha, beta = masks[:, 0:1], masks[:, 1:2] 

        # === Step B: 频域双路径交互 (振幅 + 相位) ===
        # 1. FFT 变换
        fft_rgb = torch.fft.rfft2(q_rgb, norm='backward')
        fft_ir = torch.fft.rfft2(k_ir, norm='backward')

        # 获取频谱图的实际尺寸 [B, C, H, W_freq] -> W_freq 通常是 W//2 + 1
        freq_h, freq_w = fft_rgb.shape[-2], fft_rgb.shape[-1]

        # === 新增：对齐门控权重尺寸到频域尺寸 ===
        # alpha 和 beta 原本是 [B, 1, H, W]，需要缩放到 [B, 1, H, W_freq]
        alpha_f = F.interpolate(alpha, size=(freq_h, freq_w), mode='nearest')
        beta_f = F.interpolate(beta, size=(freq_h, freq_w), mode='nearest')

        # 2. 振幅融合 (细节提纯)
        # 利用 alpha 动态融合并增强振幅
        amp_rgb, amp_ir = torch.abs(fft_rgb), torch.abs(fft_ir)
        fused_amp = alpha_f * amp_rgb + (1 - alpha_f) * amp_ir
        
        # 模拟 SK-Gate 在频域做二次增强
        amp_mask = F.interpolate(self.amp_refine(fused_amp.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).expand(-1, -1, amp_rgb.shape[-2], amp_rgb.shape[-1])), 
                                 size=(amp_rgb.shape[-2], amp_rgb.shape[-1]))
        fused_amp = fused_amp * (1 + amp_mask)

        # 3. 相位重塑 (结构对齐 - 核心创新点)
        # beta 代表 RGB 相位的可靠性，若低则引入 IR 相位修正结构
        pha_rgb, pha_ir = torch.angle(fft_rgb), torch.angle(fft_ir)
        # 通过相位插值实现结构重塑，避免硬交换导致的伪影
        fused_pha = beta_f * pha_rgb + (1 - beta_f) * pha_ir

        # 4. iFFT 还原到空域
        combined_freq = torch.polar(fused_amp, fused_pha)
        fused_spatial = torch.fft.irfft2(combined_freq, s=(h, w), norm='backward')

        # === Step C: 3D 坐标关联 (ACFA) ===
        # 进一步在还原后的空间特征上进行位置加固
        x_h = torch.mean(fused_spatial, dim=2, keepdim=True)
        x_v = torch.mean(fused_spatial, dim=3, keepdim=True)
        
        g_h = self.dw_h(F.interpolate(self.p_h, size=(h, 1), mode='bilinear', align_corners=False))
        feat_coord = (x_h * torch.sigmoid(g_h)) * x_v

        # === Step D: 残差重构 ===
        out = self.output(feat_coord)
        # 全局残差：保留原始模态的最优基础特征
        base_res = alpha * f_rgb + (1 - alpha) * f_ir
        return out + base_res


# --- 先定义我们最终确定的创新模块 SFA_MCM ---
class SFA_MCM(nn.Module):
    def __init__(self, channels, reduction=4, pool_size=7):
        super(SFA_MCM, self).__init__()
        self.inter_channels = channels // reduction
        self.pool_size = pool_size
        self.scale = self.inter_channels ** -0.5

        # 1. 频域结构引导 (处理 Step A 传来的 beta 掩码)
        self.spectral_refine = nn.Sequential(
            nn.Conv2d(1, self.inter_channels, 1), # 输入是 1 通道的 beta 掩码
            nn.Sigmoid()
        )

        # 2. 非对称轴向投影
        self.q_ir = nn.Conv2d(channels, self.inter_channels, 1)
        self.k_rgb = nn.Conv2d(channels, self.inter_channels, 1)
        self.v_fused = nn.Conv2d(channels, self.inter_channels, 1) # 投影 iFFT 的结果

        # 3. 动态 ACFA 对齐先验
        self.dynamic_offset = nn.Parameter(torch.randn(1, self.inter_channels, 32, 1) * 0.02)
        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, groups=self.inter_channels),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True)
        )

        # 4. SCA 通道关联 (Z轴)
        self.k_z = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.v_z = nn.Conv2d(self.inter_channels, self.inter_channels, 1)

    def forward(self, f_rgb, f_ir, beta, fused_spatial):
        b, c, h, w = f_rgb.shape

        # Step 1: 频域引导权重 (由相位置信度 beta 驱动)
        s_weight = self.spectral_refine(beta)
        
        # Step 2: 轴向压缩与动态对齐
        # 使用 IR 提取高度信息，RGB 提取宽度信息
        x_h_ir = torch.mean(f_ir * s_weight.expand(-1, c, -1, -1), dim=3, keepdim=True)
        x_w_rgb = torch.mean(f_rgb, dim=2, keepdim=True)

        # 注入动态偏移先验
        prior = F.interpolate(self.dynamic_offset, size=(h, 1), mode='bilinear', align_corners=False)
        x_h_aligned = x_h_ir + self.prior_conv(prior)

        # Step 3: 构建 3D 关联底座 (空间交互)
        q = self.q_ir(x_h_aligned)
        k = self.k_rgb(x_w_rgb)
        v = self.v_fused(fused_spatial) # 核心：使用 iFFT 还原后的特征作为 Value
        
        feat_spatial = (q * k) * v 

        # Step 4: SCA 跨通道语义对齐 (Z轴)
        z_q = torch.mean(s_weight, dim=(2,3), keepdim=True) # 频域结构指挥官
        # 1. 先进行空间池化保持 4D 形状: [B, C', 7, 7]
        z_feat_map = F.adaptive_avg_pool2d(feat_spatial, (self.pool_size, self.pool_size))

        # 2. 在 4D 张量上执行卷积，确保通道数(32)对齐
        z_k = self.k_z(z_feat_map) # 结果仍为 [B, C', 7, 7]

        # 3. 然后再展平用于矩阵乘法: [B, C', 49] -> transpose -> [B, 49, C']
        z_k = z_k.flatten(2).transpose(-1, -2)
        
        attn_z = torch.softmax(torch.matmul(z_q.flatten(2), z_q.flatten(2).transpose(-1, -2)) * self.scale, dim=-1)
        
        z_v = self.v_z(feat_spatial).flatten(2)
        feat_final = torch.matmul(attn_z, z_v).view(b, self.inter_channels, h, w)

        return feat_final

# --- 整合进你的 SFCA_Fusion_V2 ---
class SFCA_Fusion_V3(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.hidden = channels // reduction
        
        # --- 1. 门控与降维 ---
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden, 1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.proj_rgb = nn.Conv2d(channels, self.hidden, 1)
        self.proj_ir = nn.Conv2d(channels, self.hidden, 1)

        # --- 2. 频域处理组件 ---
        self.amp_refine = nn.Sequential(
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1, groups=self.hidden),
            nn.Sigmoid()
        )
        
        # --- 3. 空间坐标关联 (升级为 SFA-MCM) ---
        # 替换掉原来的 p_h, dw_h 等
        self.sfa_mcm = SFA_MCM(self.hidden, reduction=1) # 此处 reduction 设为 1 因为 hidden 已降维

        self.output = nn.Sequential(
            nn.Conv2d(self.hidden, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        f_rgb, f_ir = x[0], x[1]
        b, c, h, w = f_rgb.shape

        # === Step A: 门控分配 ===
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        masks = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        alpha, beta = masks[:, 0:1], masks[:, 1:2] 

        # === Step B: 频域双路径交互 ===
        fft_rgb = torch.fft.rfft2(q_rgb, norm='backward')
        fft_ir = torch.fft.rfft2(k_ir, norm='backward')
        freq_h, freq_w = fft_rgb.shape[-2], fft_rgb.shape[-1]
        alpha_f = F.interpolate(alpha, size=(freq_h, freq_w), mode='nearest')
        beta_f = F.interpolate(beta, size=(freq_h, freq_w), mode='nearest')

        # 振幅增强与相位重塑
        amp_rgb, amp_ir = torch.abs(fft_rgb), torch.abs(fft_ir)
        fused_amp = alpha_f * amp_rgb + (1 - alpha_f) * amp_ir
        # SK-Gate 增强逻辑
        amp_mask = F.interpolate(self.amp_refine(fused_amp.mean(dim=(-2,-1), keepdim=True).expand(-1, -1, freq_h, freq_w)), size=(freq_h, freq_w))
        fused_amp = fused_amp * (1 + amp_mask)

        pha_rgb, pha_ir = torch.angle(fft_rgb), torch.angle(fft_ir)
        fused_pha = beta_f * pha_rgb + (1 - beta_f) * pha_ir

        # iFFT 还原 到空域
        combined_freq = torch.polar(fused_amp, fused_pha)
        fused_spatial = torch.fft.irfft2(combined_freq, s=(h, w), norm='backward')

        # === Step C: 升级后的空间关联 (SFA-MCM) ===
        # 此处即为 SFA_MCM 的位置
        feat_coord = self.sfa_mcm(q_rgb, k_ir, beta, fused_spatial)

        # === Step D: 残差重构 ===
        out = self.output(feat_coord)
        base_res = alpha * f_rgb + (1 - alpha) * f_ir
        return out + base_res




class SpectralCrossAttention(nn.Module):
    """
    轻量化谱域交叉注意力模块
    作用：利用模态 A 的分布作为引导，从模态 B 中提取匹配的频域特征
    """
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels // 4, 1)
        self.k = nn.Conv2d(channels, channels // 4, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // 4) ** -0.5

    def forward(self, x_q, x_kv):
        b, c, h, w = x_q.shape
        # 轴向简化计算：将空间维度展平以计算相关性
        q = self.q(x_q).flatten(2) # [B, C', N]
        k = self.k(x_kv).flatten(2) # [B, C', N]
        v = self.v(x_kv).flatten(2) # [B, C, N]

        # 计算模态间相关性矩阵 [N, N]
        attn = torch.softmax(torch.matmul(q.transpose(-1, -2), k) * self.scale, dim=-1)
        
        # 语义对齐后的特征提取
        out = torch.matmul(v, attn.transpose(-1, -2))
        return out.view(b, c, h, w)

class SFCA_Fusion_V4(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.hidden = channels // reduction
        
        # 1. 门控与降维投影
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden, 1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.proj_rgb = nn.Conv2d(channels, self.hidden, 1)
        self.proj_ir = nn.Conv2d(channels, self.hidden, 1)

        # 2. 核心创新：谱域交叉注意力 (CSA)
        # 分别处理幅度 (Amplitude) 和 相位 (Phase)
        self.csa_amp = SpectralCrossAttention(self.hidden)
        self.csa_pha = SpectralCrossAttention(self.hidden)
        
        # 3. 空间域 SFA-MCM (之前定义的最终版)
        self.sfa_mcm = SFA_MCM(self.hidden, reduction=1)

        self.output = nn.Sequential(
            nn.Conv2d(self.hidden, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        f_rgb, f_ir = x[0], x[1]
        b, c, h, w = f_rgb.shape

        # === Step A: 动态门控与预处理 ===
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        masks = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        alpha, beta = masks[:, 0:1], masks[:, 1:2] 

        # === Step B: 频域交叉注意力交互 (CSA) ===
        # 1. FFT 变换
        fft_rgb = torch.fft.rfft2(q_rgb, norm='backward')
        fft_ir = torch.fft.rfft2(k_ir, norm='backward')

        # 获取频域的实际尺寸
        freq_h, freq_w = fft_rgb.shape[-2], fft_rgb.shape[-1] # 例如 32, 17
        
        # 【重要修复】：将 alpha 和 beta 缩放到频域尺寸
        alpha_f = F.interpolate(alpha, size=(freq_h, freq_w), mode='nearest')
        beta_f = F.interpolate(beta, size=(freq_h, freq_w), mode='nearest')
        
        # 分离幅度和相位
        amp_rgb, amp_ir = torch.abs(fft_rgb), torch.abs(fft_ir)
        pha_rgb, pha_ir = torch.angle(fft_rgb), torch.angle(fft_ir)

        # 2. 幅度域交叉注意力增强 (FCA++)
        # 以 RGB 为 Query，从 IR 中检索能量补充
        amp_fused = self.csa_amp(amp_rgb, amp_ir)
        # 结合 alpha 掩码进行残差融合
        amp_final = alpha_f * amp_rgb + (1 - alpha_f) * amp_fused

        # 3. 相位域交叉注意力重塑 (Phase-Align++)
        # 以 IR 的结构为 Query，从 RGB 中检索边缘细节进行精准对齐
        # 这解决了硬插值导致的重影问题
        pha_fused = self.csa_pha(pha_ir, pha_rgb) 
        # 结合 beta 掩码决定最终结构
        pha_final = beta_f * pha_rgb + (1 - beta_f) * pha_fused

        # 4. iFFT 还原
        combined_freq = torch.polar(amp_final, pha_final)
        fused_spatial = torch.fft.irfft2(combined_freq, s=(h, w), norm='backward')

        # === Step C: 空间域 3D 关联 (SFA-MCM) ===
        feat_coord = self.sfa_mcm(q_rgb, k_ir, beta, fused_spatial)

        # === Step D: 残差重构 ===
        out = self.output(feat_coord)
        base_res = alpha * f_rgb + (1 - alpha) * f_ir
        return out + base_res
    


# --- 1. 谱域交叉注意力模块 (CSA) ---
class SpectralCrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels // 4, 1)
        self.k = nn.Conv2d(channels, channels // 4, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // 4) ** -0.5

    def forward(self, x_q, x_kv):
        b, c, h, w = x_q.shape
        # 展平空间维度进行谱域交互
        q = self.q(x_q).flatten(2) 
        k = self.k(x_kv).flatten(2) 
        v = self.v(x_kv).flatten(2) 

        attn = torch.softmax(torch.matmul(q.transpose(-1, -2), k) * self.scale, dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2))
        return out.view(b, c, h, w)

# --- 2. 增强型非对称魔方关联模块 (SFA-MCM-Decoupled) ---
class SFA_MCM_Decoupled(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.inter_channels = channels // reduction
        
        # Head A: IR(H) x RGB(W) - 偏向横向目标
        self.q_h_a = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.k_w_a = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        
        # Head B: RGB(H) x IR(W) - 偏向纵向目标
        self.q_h_b = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.k_w_b = nn.Conv2d(self.inter_channels, self.inter_channels, 1)

        # 动态 ACFA 先验 (Dynamic Position Prior)
        self.offset_h = nn.Parameter(torch.randn(1, self.inter_channels, 32, 1) * 0.02)
        self.offset_w = nn.Parameter(torch.randn(1, self.inter_channels, 1, 32) * 0.02)
        
        # 聚合组件
        self.v_proj = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.head_fusion = nn.Conv2d(self.inter_channels * 2, self.inter_channels, 1)
        self.out_proj = nn.Conv2d(self.inter_channels, channels, 1)

    def forward(self, f_rgb, f_ir, fused_spatial):
        b, c, h, w = f_rgb.shape # 此时输入的 f_rgb, f_ir 已是 hidden 维度

        # 轴向压缩
        ir_h, ir_w = torch.mean(f_ir, dim=3, keepdim=True), torch.mean(f_ir, dim=2, keepdim=True)
        rgb_h, rgb_w = torch.mean(f_rgb, dim=3, keepdim=True), torch.mean(f_rgb, dim=2, keepdim=True)

        # 注入动态偏移
        off_h = F.interpolate(self.offset_h, size=(h, 1), mode='bilinear', align_corners=False)
        off_w = F.interpolate(self.offset_w, size=(1, w), mode='bilinear', align_corners=False)
        
        # 双头非对称交互 (消除方向偏见)
        head_a = self.q_h_a(ir_h + off_h) * self.k_w_a(rgb_w + off_w)
        head_b = self.q_h_b(rgb_h + off_h) * self.k_w_b(ir_w + off_w)

        # 响应融合
        spatial_weight = torch.sigmoid(self.head_fusion(torch.cat([head_a, head_b], dim=1)))
        
        v = self.v_proj(fused_spatial)
        return self.out_proj(spatial_weight * v)

# --- 3. 顶层融合模块 (SFCA-Fusion V5) ---
class SFCA_Fusion_V5(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.hidden = channels // reduction
        
        # 1. 动态门控
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden, 1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.proj_rgb = nn.Conv2d(channels, self.hidden, 1)
        self.proj_ir = nn.Conv2d(channels, self.hidden, 1)

        # 2. 谱域交叉注意力 (CSA)
        self.csa_amp = SpectralCrossAttention(self.hidden)
        self.csa_pha = SpectralCrossAttention(self.hidden)
        
        # 3. 旋转鲁棒空间关联 (SFA-MCM)
        # self.sfa_mcm = SFA_MCM_Decoupled(self.hidden, reduction=1)
        self.cross_mcm = CrossModalMagicCubeModule(self.hidden, reduction=1) 

        self.output = nn.Sequential(
            nn.Conv2d(self.hidden, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        f_rgb, f_ir = x[0], x[1]
        b, c, h, w = f_rgb.shape

        # Step A: 投影与门控
        q_rgb, k_ir = self.proj_rgb(f_rgb), self.proj_ir(f_ir)
        masks = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        alpha, beta = masks[:, 0:1], masks[:, 1:2] 

        # Step B: 谱域处理 (CSA)
        fft_rgb = torch.fft.rfft2(q_rgb, norm='backward')
        fft_ir = torch.fft.rfft2(k_ir, norm='backward')
        
        # 解决维度匹配的关键：掩码重采样到频域尺寸
        fh, fw = fft_rgb.shape[-2], fft_rgb.shape[-1]
        alpha_f = F.interpolate(alpha, size=(fh, fw), mode='nearest')
        beta_f = F.interpolate(beta, size=(fh, fw), mode='nearest')

        # 幅度/相位分离与 CSA 检索
        amp_rgb, amp_ir = torch.abs(fft_rgb), torch.abs(fft_ir)
        pha_rgb, pha_ir = torch.angle(fft_rgb), torch.angle(fft_ir)

        amp_fused = alpha_f * amp_rgb + (1 - alpha_f) * self.csa_amp(amp_rgb, amp_ir)
        pha_fused = beta_f * pha_rgb + (1 - beta_f) * self.csa_pha(pha_ir, pha_rgb)

        # iFFT 还原
        fused_spatial = torch.fft.irfft2(torch.polar(amp_fused, pha_fused), s=(h, w), norm='backward')

        # Step C: 空间对齐 (多头解耦)
        feat_coord = self.sfa_mcm(q_rgb, k_ir, fused_spatial)

        # Step D: 残差重构
        out = self.output(feat_coord)
        base_res = alpha * f_rgb + (1 - alpha) * f_ir
        return out + base_res


class SFA_MCM_Decoupled_BetaGuided(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.inter_channels = channels // reduction
        
        # 频域结构引导 - 恢复beta的使用
        self.spectral_refine = nn.Sequential(
            nn.Conv2d(1, self.inter_channels, 1),
            nn.Sigmoid()
        )
        
        # Head A: IR(H) x RGB(W)
        self.q_h_a = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.k_w_a = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        
        # Head B: RGB(H) x IR(W)
        self.q_h_b = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.k_w_b = nn.Conv2d(self.inter_channels, self.inter_channels, 1)

        # 动态 ACFA 先验
        self.offset_h = nn.Parameter(torch.randn(1, self.inter_channels, 32, 1) * 0.02)
        self.offset_w = nn.Parameter(torch.randn(1, self.inter_channels, 1, 32) * 0.02)
        
        # 聚合组件
        self.v_proj = nn.Conv2d(self.inter_channels, self.inter_channels, 1)
        self.head_fusion = nn.Conv2d(self.inter_channels * 2, self.inter_channels, 1)
        self.out_proj = nn.Conv2d(self.inter_channels, channels, 1)
        
        # 简化版通道关联：使用SE-like机制
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inter_channels, self.inter_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels // 4, self.inter_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, f_rgb, f_ir, beta, fused_spatial):
        b, c, h, w = f_rgb.shape

        # Step 1: 频域引导权重
        s_weight = self.spectral_refine(beta)
        
        # 轴向压缩 - 使用beta引导调制
        ir_weighted = f_ir * s_weight.expand(-1, c, -1, -1)
        ir_h, ir_w = torch.mean(ir_weighted, dim=3, keepdim=True), torch.mean(f_ir, dim=2, keepdim=True)
        rgb_h, rgb_w = torch.mean(f_rgb, dim=3, keepdim=True), torch.mean(f_rgb, dim=2, keepdim=True)

        # 注入动态偏移
        off_h = F.interpolate(self.offset_h, size=(h, 1), mode='bilinear', align_corners=False)
        off_w = F.interpolate(self.offset_w, size=(1, w), mode='bilinear', align_corners=False)
        
        # 双头非对称交互
        head_a = self.q_h_a(ir_h + off_h) * self.k_w_a(rgb_w + off_w)
        head_b = self.q_h_b(rgb_h + off_h) * self.k_w_b(ir_w + off_w)

        # 响应融合
        spatial_weight = torch.sigmoid(self.head_fusion(torch.cat([head_a, head_b], dim=1)))
        
        v = self.v_proj(fused_spatial)
        feat_spatial = spatial_weight * v
        
        # === 简化版通道关联：SE-style ===
        channel_weight = self.se(feat_spatial)  # [B, C, 1, 1]
        feat_channel = feat_spatial * channel_weight
        
        # 融合（可选：使用beta调制通道权重）
        beta_channel = F.adaptive_avg_pool2d(beta, 1)
        feat_final = feat_spatial + 0.1 * feat_channel * beta_channel

        return self.out_proj(feat_final)

        
# --- 3. 顶层融合模块 (SFCA-Fusion V5 with Beta Guidance) ---
class SFCA_Fusion_V6(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.hidden = channels // reduction
        
        # 1. 动态门控
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden, 1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.proj_rgb = nn.Conv2d(channels, self.hidden, 1)
        self.proj_ir = nn.Conv2d(channels, self.hidden, 1)

        # 2. 谱域交叉注意力 (CSA) - 保持V5的设计
        self.csa_amp = SpectralCrossAttention(self.hidden)
        self.csa_pha = SpectralCrossAttention(self.hidden)
        
        # 3. 旋转鲁棒空间关联 - 使用恢复beta引导的新模块
        self.sfa_mcm = SFA_MCM_Decoupled_BetaGuided(self.hidden, reduction=1)

        self.output = nn.Sequential(
            nn.Conv2d(self.hidden, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        f_rgb, f_ir = x[0], x[1]
        b, c, h, w = f_rgb.shape

        # Step A: 投影与门控
        q_rgb, k_ir = self.proj_rgb(f_rgb), self.proj_ir(f_ir)
        masks = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        alpha, beta = masks[:, 0:1], masks[:, 1:2] 

        # Step B: 谱域处理 (CSA)
        fft_rgb = torch.fft.rfft2(q_rgb, norm='backward')
        fft_ir = torch.fft.rfft2(k_ir, norm='backward')
        
        fh, fw = fft_rgb.shape[-2], fft_rgb.shape[-1]
        alpha_f = F.interpolate(alpha, size=(fh, fw), mode='nearest')
        beta_f = F.interpolate(beta, size=(fh, fw), mode='nearest')

        # 幅度/相位分离与 CSA 检索
        amp_rgb, amp_ir = torch.abs(fft_rgb), torch.abs(fft_ir)
        pha_rgb, pha_ir = torch.angle(fft_rgb), torch.angle(fft_ir)

        amp_fused = alpha_f * amp_rgb + (1 - alpha_f) * self.csa_amp(amp_rgb, amp_ir)
        pha_fused = beta_f * pha_rgb + (1 - beta_f) * self.csa_pha(pha_ir, pha_rgb)

        # iFFT 还原
        fused_spatial = torch.fft.irfft2(torch.polar(amp_fused, pha_fused), s=(h, w), norm='backward')

        # Step C: 空间对齐 - 传入beta进行引导
        feat_coord = self.sfa_mcm(q_rgb, k_ir, beta, fused_spatial)

        # Step D: 残差重构
        out = self.output(feat_coord)
        base_res = alpha * f_rgb + (1 - alpha) * f_ir
        return out + base_res


# -------------------------------------------------------------------------------------------------------------------

class MultiScaleDoG(nn.Module):
    """
    多尺度高斯差分模块 (Pyramid Difference of Gaussians, P-DoG)
    
    提取符合生物视觉特性的多尺度细节分量
    对应论文公式(1)和(2)
    """
    def __init__(self, channels, scales=[1, 2, 4, 8], k=1.6):
        super().__init__()
        self.scales = scales
        self.k = k
        self.channels = channels
        
        # 预计算不同尺度的高斯核
        self.gaussians = nn.ModuleList()
        for sigma in scales:
            self.gaussians.append(GaussianBlur(channels, sigma))
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            details: 多尺度细节分量融合结果 [B, C, H, W]
        """
        b, c, h, w = x.shape
        all_details = []
        
        for i, sigma in enumerate(self.scales):
            # 计算 G(kσ) * F 和 G(σ) * F
            g_k = self.gaussians[i](x)
            g_sigma = self.gaussians[i](x) if i == 0 else self.gaussians[i-1](x)
            
            # 差分操作: D = [G(kσ) - G(σ)] * F
            detail = g_k - g_sigma
            all_details.append(detail)
        
        # 多尺度细节融合 (求和)
        fused_detail = torch.stack(all_details, dim=0).sum(dim=0)
        
        return fused_detail


class GaussianBlur(nn.Module):
    """
    高斯模糊层
    用于实现高斯差分操作
    """
    def __init__(self, channels, sigma):
        super().__init__()
        self.sigma = sigma
        self.channels = channels
        self.kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        
        # 生成高斯核
        self.register_buffer('kernel', self._get_gaussian_kernel())
        
    def _get_gaussian_kernel(self):
        """生成2D高斯核"""
        kernel_size = self.kernel_size
        sigma = self.sigma
        
        # 创建坐标网格
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # 计算高斯分布
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # 扩展为适用于卷积的维度 [1, 1, K, K]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        return kernel
    
    def forward(self, x):
        """应用高斯模糊"""
        if self.kernel_size > 1:
            # 对每个通道分别应用高斯模糊
            b, c, h, w = x.shape
            kernel = self.kernel.repeat(c, 1, 1, 1)
            x_blurred = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=c)
            return x_blurred
        return x

class LightweightDoGFilter(nn.Module):
    """
    轻量级高斯差分滤波器
    使用分离卷积和可学习参数
    """
    def __init__(self, channels, sigma1=1.0, sigma2=2.0):
        super().__init__()
        self.channels = channels
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        # 生成高斯核
        kernel1 = self._gaussian_kernel(sigma1)
        kernel2 = self._gaussian_kernel(sigma2)
        
        # 注册为buffer（不参与训练）
        self.register_buffer('kernel1', kernel1)
        self.register_buffer('kernel2', kernel2)

        self.channel_weight = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.ones_(self.channel_weight.weight)
        
    def _gaussian_kernel(self, sigma):
        """生成2D高斯核"""
        size = int(2 * np.ceil(3 * sigma) + 1)
        if size % 2 == 0:
            size += 1
        
        ax = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, size, size)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 扩展核到所有通道
        kernel1 = self.kernel1.repeat(c, 1, 1, 1)
        kernel2 = self.kernel2.repeat(c, 1, 1, 1)
        
        # 应用高斯滤波
        size1 = self.kernel1.shape[-1]
        size2 = self.kernel2.shape[-1]
        
        blurred1 = F.conv2d(x, kernel1, padding=size1//2, groups=c)
        blurred2 = F.conv2d(x, kernel2, padding=size2//2, groups=c)
        
        # 高斯差分
        dog = blurred1 - blurred2
        
        # 通道自适应加权
        dog = self.channel_weight(dog)
        
        return dog
    
class JointTextureReasoning(nn.Module):
    """
    多尺度联合纹理推理模块
    
    通过Sobel梯度提取和共识纹理图定位跨模态共同边缘区域
    对应论文公式(3): Γ = Sobel(D_rgb ∪ D_ir) ⊖ (D_rgb ∩ D_ir)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Sobel算子 (水平方向和垂直方向)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def _sobel_gradient(self, x):
        """计算Sobel梯度幅值"""
        b, c, h, w = x.shape
        
        # 扩展Sobel算子到所有通道
        sobel_x = self.sobel_x.repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1)
        
        # 计算梯度
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=c)
        
        # 梯度幅值
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return grad_mag
    
    def forward(self, d_rgb, d_ir):
        """
        Args:
            d_rgb: RGB细节分量 [B, C, H, W]
            d_ir: 红外细节分量 [B, C, H, W]
        Returns:
            gamma: 独特边缘增强因子 [B, C, H, W]
        """
        # 计算Sobel梯度
        grad_rgb = self._sobel_gradient(d_rgb)
        grad_ir = self._sobel_gradient(d_ir)
        
        # 并集操作 (取最大值，表示任一模态有边缘)
        union = torch.max(grad_rgb, grad_ir)
        
        # 交集操作 (取最小值，表示两模态都有边缘)
        intersection = torch.min(grad_rgb, grad_ir)
        
        # 冗余剔除: Γ = Sobel(union) ⊖ intersection
        # 这里使用差值操作剔除共性冗余
        gamma = union - intersection
        gamma = torch.clamp(gamma, min=0)  # 确保非负
        
        # 特征融合增强
        gamma = self.fusion(torch.cat([gamma, gamma], dim=1))
        
        return gamma


class IlluminationAwareGate(nn.Module):
    """
    亮度感知门控模块 (Illumination-Aware Gate, IAG)
    
    根据环境光照动态调整RGB和红外模态的互补策略
    对应论文公式(4): L_env = Sigmoid(MLP(GAP(F_rgb)))
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 1),
            nn.Sigmoid()
        )
        
    def forward(self, f_rgb):
        """
        Args:
            f_rgb: RGB特征 [B, C, H, W]
        Returns:
            L_env: 环境光照系数 [B, 1, 1, 1], 0表示暗光, 1表示强光
        """
        b, c, h, w = f_rgb.shape
        
        # 全局平均池化
        pooled = self.gap(f_rgb).view(b, c)  # [B, C]
        
        # MLP映射得到光照系数
        L_env = self.mlp(pooled).view(b, 1, 1, 1)  # [B, 1, 1, 1]
        
        return L_env


class AsymmetricCrossModalInteraction(nn.Module):
    """
    非对称跨模态交互模块
    
    根据光照条件动态注入互补特征
    对应论文公式(5)和(6):
    F_rgb' = F_rgb + (1 - L_env) * (Gate_ir(F_ir_low) ⊙ F_ir_low)
    F_ir' = F_ir + L_env * (Gate_rgb(F_rgb_high) ⊙ F_rgb_high)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # IR低维结构骨架提取 (暗光下注入RGB)
        self.ir_low_extractor = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        
        # RGB高维纹理提取 (强光下注入IR)
        self.rgb_high_extractor = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        
        # 门控函数 Gate_ir 和 Gate_rgb
        # 用于选择性激活注入的特征
        self.gate_ir = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.gate_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 可学习缩放因子 (零初始化)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))
        
    def forward(self, f_rgb, f_ir, L_env):
        """
        Args:
            f_rgb: RGB特征 [B, C, H, W]
            f_ir: 红外特征 [B, C, H, W]
            L_env: 环境光照系数 [B, 1, 1, 1]
        Returns:
            f_rgb_enhanced: 增强后的RGB特征 [B, C, H, W]
            f_ir_enhanced: 增强后的红外特征 [B, C, H, W]
        """
        # 提取IR低维结构骨架
        f_ir_low = self.ir_low_extractor(f_ir)
        
        # 提取RGB高维纹理
        f_rgb_high = self.rgb_high_extractor(f_rgb)
        
        # 计算门控权重
        gate_ir_weight = self.gate_ir(f_ir_low)  # [B, C, H, W]
        gate_rgb_weight = self.gate_rgb(f_rgb_high)  # [B, C, H, W]
        
        # 非对称交互
        # 暗光下 (L_env → 0): 注入IR结构骨架到RGB
        f_rgb_enhanced = f_rgb + self.gamma_rgb * (1 - L_env) * (gate_ir_weight * f_ir_low)
        
        # 强光下 (L_env → 1): 注入RGB纹理到IR
        f_ir_enhanced = f_ir + self.gamma_ir * L_env * (gate_rgb_weight * f_rgb_high)
        
        return f_rgb_enhanced, f_ir_enhanced


class TIGE(nn.Module):
    """
    纹理-光照双引导特征增强模块 (Texture-Illumination Dual-Guided Feature Enhancement)
    
    完整的TIGE模块,整合所有子模块:
    1. 多尺度联合纹理推理 (JointTextureReasoning)
    2. 亮度感知门控 (IlluminationAwareGate)
    3. 非对称跨模态交互 (AsymmetricCrossModalInteraction)
    4. 零初始化残差约束 (Zero-Init Residual Constraint)
    
    对应论文第3.2节
    """
    def __init__(self, channels, reduction=4, scales=[1, 2, 4, 8],):
        super().__init__()
        self.channels = channels
        
        # 1. 多尺度高斯差分 (P-DoG)
        self.dog_rgb = LightweightDoGFilter(channels)
        self.dog_ir = LightweightDoGFilter(channels)
        
        # 2. 联合纹理推理 (Joint Texture Reasoning)
        self.texture_reasoning = JointTextureReasoning(channels)
        
        # 3. 亮度感知门控 (Illumination-Aware Gate)
        self.illumination_gate = IlluminationAwareGate(channels, reduction)
        
        # 4. 非对称跨模态交互 (Asymmetric Cross-Modal Interaction)
        self.asymmetric_interaction = AsymmetricCrossModalInteraction(channels, reduction)
        
        # 5. 零初始化残差约束 (Zero-Init Residual Constraint)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.layer_norm = nn.LayerNorm(channels)
        
        # 辅助：特征投影层
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, f_rgb, f_ir):
        """
        Args:
            f_rgb: RGB特征 [B, C, H, W]
            f_ir: 红外特征 [B, C, H, W]
        Returns:
            out_rgb: 增强后的RGB特征 [B, C, H, W]
            out_ir: 增强后的红外特征 [B, C, H, W]
        """
        b, c, h, w = f_rgb.shape
        
        # ========== Step 1: 多尺度细节提取 ==========
        # 对应论文公式(1)和(2)
        d_rgb = self.dog_rgb(f_rgb)  # RGB细节分量
        d_ir = self.dog_ir(f_ir)      # IR细节分量
        
        # ========== Step 2: 联合纹理推理 ==========
        # 对应论文公式(3)
        # 生成独特边缘增强因子 Γ
        gamma_texture = self.texture_reasoning(d_rgb, d_ir)  # [B, C, H, W]
        
        # 应用纹理增强因子到原始特征
        f_rgb_texture_enhanced = f_rgb + gamma_texture * d_rgb
        f_ir_texture_enhanced = f_ir + gamma_texture * d_ir
        
        # ========== Step 3: 亮度感知门控 ==========
        # 对应论文公式(4)
        L_env = self.illumination_gate(f_rgb)  # [B, 1, 1, 1]
        
        # ========== Step 4: 非对称跨模态交互 ==========
        # 对应论文公式(5)和(6)
        f_rgb_interacted, f_ir_interacted = self.asymmetric_interaction(
            f_rgb_texture_enhanced, f_ir_texture_enhanced, L_env
        )
        
        # ========== Step 5: 零初始化残差约束 ==========
        # 对应论文公式(7)
        # 对RGB分支
        residual_rgb = f_rgb + self.gamma * f_rgb_interacted
        out_rgb = self.layer_norm(residual_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 对IR分支
        residual_ir = f_ir + self.gamma * f_ir_interacted
        out_ir = self.layer_norm(residual_ir.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return out_rgb, out_ir
    

class SpectralCrossAttention(nn.Module):
    """
    轻量化谱域交叉注意力模块
    作用：利用模态 A 的分布作为引导，从模态 B 中提取匹配的频域特征
    """
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels // 4, 1)
        self.k = nn.Conv2d(channels, channels // 4, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // 4) ** -0.5

    def forward(self, x_q, x_kv):
        b, c, h, w = x_q.shape
        # 轴向简化计算：将空间维度展平以计算相关性
        q = self.q(x_q).flatten(2) # [B, C', N]
        k = self.k(x_kv).flatten(2) # [B, C', N]
        v = self.v(x_kv).flatten(2) # [B, C, N]

        # 计算模态间相关性矩阵 [N, N]
        attn = torch.softmax(torch.matmul(q.transpose(-1, -2), k) * self.scale, dim=-1)
        
        # 语义对齐后的特征提取
        out = torch.matmul(v, attn.transpose(-1, -2))
        return out.view(b, c, h, w)


class CrossModalMagicCubeModule(nn.Module):
    """
    跨模态魔方模块 (Cross-Modal Magic Cube Module)
    
    将OSIV-Net中的MCM扩展到跨模态场景，实现RGB和红外特征的三维关联建模。
    
    核心设计：
    1. 轴向特征解耦：分别沿H轴和W轴进行池化，捕获长程依赖
    2. 跨模态交互：在水平和垂直方向上分别建立RGB与IR的关联
    3. 通道关联：在空间建模后，进一步沿通道维度整合信息
    4. 动态偏移先验：可学习的轴向偏移，适配无人机视差
    
    参考：OSIV-Net (Zhang et al., ISPRS 2025) 中的 Magic Cube Module
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.inter_channels = channels // reduction
        
        # ========== 1. 轴向特征投影 ==========
        # 将RGB和IR特征投影到交互空间
        self.proj_rgb = nn.Conv2d(channels, self.inter_channels, 1)
        self.proj_ir = nn.Conv2d(channels, self.inter_channels, 1)
        
        # ========== 2. 轴向注意力 (模仿MCM的x和y轴处理) ==========
        # 水平方向注意力 (沿H轴) - 对应MCM的y轴方向
        self.h_attn = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, self.inter_channels, 1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.inter_channels, 1),
            nn.Sigmoid()
        )
        
        # 垂直方向注意力 (沿W轴) - 对应MCM的x轴方向
        self.w_attn = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, self.inter_channels, 1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.inter_channels, 1),
            nn.Sigmoid()
        )
        
        # ========== 3. 通道关联模块 (模仿MCM的z轴处理) ==========
        # 在空间建模后，沿通道维度建立关联
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inter_channels, self.inter_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels // 4, self.inter_channels, 1),
            nn.Sigmoid()
        )
        
        # ========== 4. 动态轴向偏移先验 (适配无人机视差) ==========
        # 可学习的偏移量，模拟传感器视差和振动
        self.offset_h = nn.Parameter(torch.randn(1, self.inter_channels, 32, 1) * 0.02)
        self.offset_w = nn.Parameter(torch.randn(1, self.inter_channels, 1, 32) * 0.02)
        
        # ========== 5. 特征融合与输出 ==========
        self.fusion = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, channels, 1)
        )
        
        # 可学习残差系数
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, f_rgb, f_ir, fused_spatial=None):
        """
        Args:
            f_rgb: RGB特征 [B, C, H, W]
            f_ir: 红外特征 [B, C, H, W]
            fused_spatial: 频域融合后的空间特征 [B, C, H, W] (可选)
        Returns:
            out: 增强后的特征 [B, C, H, W]
        """
        b, c, h, w = f_rgb.shape
        
        # Step 1: 投影到交互空间
        q_rgb = self.proj_rgb(f_rgb)      # [B, C', H, W]
        k_ir = self.proj_ir(f_ir)         # [B, C', H, W]
        
        # Step 2: 动态偏移注入 (模仿MCM中对轴向的灵活建模)
        off_h = F.interpolate(self.offset_h, size=(h, 1), mode='bilinear', align_corners=False)
        off_w = F.interpolate(self.offset_w, size=(1, w), mode='bilinear', align_corners=False)
        q_rgb_shifted = q_rgb + off_h + off_w
        k_ir_shifted = k_ir + off_h + off_w
        
        # Step 3: 轴向特征解耦 (仿照MCM的x和y轴池化)
        # 水平方向 (沿H轴平均池化，得到 [B, C', 1, W]) - 对应MCM的y轴
        q_rgb_h = torch.mean(q_rgb_shifted, dim=2, keepdim=True)   # [B, C', 1, W]
        k_ir_h = torch.mean(k_ir_shifted, dim=2, keepdim=True)      # [B, C', 1, W]
        
        # 垂直方向 (沿W轴平均池化，得到 [B, C', H, 1]) - 对应MCM的x轴
        q_rgb_w = torch.mean(q_rgb_shifted, dim=3, keepdim=True)   # [B, C', H, 1]
        k_ir_w = torch.mean(k_ir_shifted, dim=3, keepdim=True)      # [B, C', H, 1]
        
        # Step 4: 跨模态轴向交互
        # 水平方向: 将RGB和IR的水平特征拼接后生成注意力权重
        h_concat = torch.cat([q_rgb_h, k_ir_h], dim=1)   # [B, 2*C', 1, W]
        h_attn_weight = self.h_attn(h_concat)             # [B, C', 1, W]
        
        # 垂直方向: 将RGB和IR的垂直特征拼接后生成注意力权重
        w_concat = torch.cat([q_rgb_w, k_ir_w], dim=1)   # [B, 2*C', H, 1]
        w_attn_weight = self.w_attn(w_concat)             # [B, C', H, 1]
        
        # Step 5: 应用轴向注意力 (仿照MCM的空间信息聚合)
        # 水平注意力沿H维扩展，垂直注意力沿W维扩展
        h_attn_expanded = h_attn_weight.expand(-1, -1, h, -1)   # [B, C', H, W]
        w_attn_expanded = w_attn_weight.expand(-1, -1, -1, w)   # [B, C', H, W]
        
        # 融合轴向注意力 (仿照MCM的乘法聚合)
        # spatial_attn = torch.sigmoid(h_attn_expanded * w_attn_expanded)  # [B, C', H, W]
        spatial_attn = h_attn_expanded * w_attn_expanded  # [B, C', H, W]
        
        # Step 6: 应用空间注意力到融合特征
        if fused_spatial is not None:
            # 如果有频域融合后的特征，使用投影后的特征
            feat_for_attn = self.proj_rgb(fused_spatial)
        else:
            # 否则使用RGB和IR的加权融合
            feat_for_attn = q_rgb + k_ir
        
        # 空间注意力增强
        feat_spatial = feat_for_attn * spatial_attn
        
        # Step 7: 通道关联 (仿照MCM的z轴处理)
        # 在空间建模后，进一步沿通道维度建立关联
        channel_weight = self.channel_attn(feat_spatial)   # [B, C', 1, 1]
        feat_channel = feat_spatial * channel_weight
        
        # Step 8: 特征融合
        # 融合空间和通道增强的特征
        feat_enhanced = feat_spatial + 0.1 * feat_channel
        
        # Step 9: 输出投影与残差连接
        out = self.fusion(feat_enhanced)
        
        # 残差连接: 保留原始特征的同时加入增强特征
        if fused_spatial is not None:
            # 如果有频域融合特征，使用其作为残差基础
            out = fused_spatial + self.gamma * out
        else:
            # 否则使用RGB和IR的简单融合作为残差基础
            base_res = (f_rgb + f_ir) / 2
            out = base_res + self.gamma * out
            
        return out


class SSCF(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.hidden = channels // reduction
        
        # 1. 动态门控
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden, 1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.proj_rgb = nn.Conv2d(channels, self.hidden, 1)
        self.proj_ir = nn.Conv2d(channels, self.hidden, 1)

        # 2. 谱域交叉注意力 (CSA)
        self.csa_amp = SpectralCrossAttention(self.hidden)
        self.csa_pha = SpectralCrossAttention(self.hidden)
        
        # 3. 旋转鲁棒空间关联 (SFA-MCM)
        # self.sfa_mcm = SFA_MCM_Decoupled(self.hidden, reduction=1)
        self.cross_mcm = CrossModalMagicCubeModule(self.hidden, reduction=1) 

        self.output = nn.Sequential(
            nn.Conv2d(self.hidden, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        f_rgb, f_ir = x[0], x[1]
        b, c, h, w = f_rgb.shape

        # Step A: 投影与门控
        q_rgb, k_ir = self.proj_rgb(f_rgb), self.proj_ir(f_ir)
        masks = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        alpha, beta = masks[:, 0:1], masks[:, 1:2] 

        # Step B: 谱域处理 (CSA)
        fft_rgb = torch.fft.rfft2(q_rgb, norm='backward')
        fft_ir = torch.fft.rfft2(k_ir, norm='backward')
        
        # 解决维度匹配的关键：掩码重采样到频域尺寸
        fh, fw = fft_rgb.shape[-2], fft_rgb.shape[-1]
        alpha_f = F.interpolate(alpha, size=(fh, fw), mode='nearest')
        beta_f = F.interpolate(beta, size=(fh, fw), mode='nearest')

        # 幅度/相位分离与 CSA 检索
        amp_rgb, amp_ir = torch.abs(fft_rgb), torch.abs(fft_ir)
        pha_rgb, pha_ir = torch.angle(fft_rgb), torch.angle(fft_ir)

        amp_fused = alpha_f * amp_rgb + (1 - alpha_f) * self.csa_amp(amp_rgb, amp_ir)
        pha_fused = beta_f * pha_rgb + (1 - beta_f) * self.csa_pha(pha_ir, pha_rgb)

        # iFFT 还原
        fused_spatial = torch.fft.irfft2(torch.polar(amp_fused, pha_fused), s=(h, w), norm='backward')

        # Step C: 空间对齐 (多头解耦)
        feat_coord = self.cross_mcm(q_rgb, k_ir, fused_spatial)

        # Step D: 残差重构
        out = self.output(feat_coord)
        base_res = alpha * f_rgb + (1 - alpha) * f_ir
        return out + base_res




class CAFusion_base(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, channels, reduction=16):
        super().__init__()

    def forward(self, x1, x2=None):
        if x2 is not None:
            x_rgb, x_ir = x1, x2
        else:
            x_rgb, x_ir = x1[0], x1[1]
        return x_rgb, x_ir
    

##-------------------------------------------------
class ContentGuidedEnhancementv2(nn.Module):
    """内容引导增强 - 优化内存版本"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        
        # 使用简化的交叉注意力（特征图小时才启用）
        self.use_attention = True
        self.channels = channels
        self.reduction = reduction
        
        # 深度卷积特征提取
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # 简化的交叉注意力（低分辨率特征图使用）
        r = max(channels // reduction, 16)
        self.query = nn.Conv2d(channels, r, 1)
        self.key = nn.Conv2d(channels, r, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, f_main, f_aux, content_map):
        B, C, H, W = f_main.shape
        
        # 内容加权
        f_main_weighted = f_main * content_map
        f_aux_weighted = f_aux * content_map
        
        # 如果特征图太大，使用深度卷积而非注意力
        if H * W > 16384:  # 大于128x128
            # 使用深度卷积
            f_main_enhanced = self.depthwise_conv(f_main_weighted)
            f_aux_enhanced = self.depthwise_conv(f_aux_weighted)
            x = f_main_enhanced + f_aux_enhanced
        else:
            # 使用交叉注意力
            # 降采样以减少计算量
            if H * W > 4096:  # 64x64以上时降采样
                scale = 2
                f_main_small = F.avg_pool2d(f_main_weighted, scale)
                f_aux_small = F.avg_pool2d(f_aux_weighted, scale)
                _, _, Hs, Ws = f_main_small.shape
                
                Q = self.query(f_main_small).view(B, -1, Hs*Ws)
                K = self.key(f_aux_small).view(B, -1, Hs*Ws)
                V = self.value(f_aux_small).view(B, C, Hs*Ws)
                
                attn = F.softmax(Q.transpose(-2, -1) @ K / (Q.shape[1] ** 0.5), dim=-1)
                x_small = (V @ attn).view(B, C, Hs, Ws)
                x = F.interpolate(x_small, size=(H, W), mode='bilinear')
            else:
                Q = self.query(f_main_weighted).view(B, -1, H*W)
                K = self.key(f_aux_weighted).view(B, -1, H*W)
                V = self.value(f_aux_weighted).view(B, C, H*W)
                
                attn = F.softmax(Q.transpose(-2, -1) @ K / (Q.shape[1] ** 0.5), dim=-1)
                x = (V @ attn).view(B, C, H, W)
        
        # 融合
        out = self.fusion(torch.cat([f_main, self.gamma * x], dim=1))
        
        return out
    
class TIGEv5(nn.Module):
    def __init__(self, channels, reduction=4, num_heads=4, window_size=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.reduction = reduction
        self.window_size = window_size
        
        # ========== 1. 纹理引导卷积（替代注意力，避免OOM）==========
        # 使用深度可分离卷积 + 通道注意力，比窗口注意力更省内存
        self.texture_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            # 通道注意力
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.texture_ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1)
        )
        
        # ========== 2. 内容引导增强 ==========
        self.content_enhance = ContentGuidedEnhancementv2(channels, reduction)
        
        # ========== 3. 光照感知 ==========
        self.illumination_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.GELU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # ========== 4. 跨模态交互 ==========
        self.interact_ir = nn.Conv2d(channels, channels, 1)
        self.interact_rgb = nn.Conv2d(channels, channels, 1)
        
        # ========== 5. 归一化 ==========
        self.norm_rgb = nn.BatchNorm2d(channels)
        self.norm_ir = nn.BatchNorm2d(channels)
        
        self.res_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, f_rgb, f_ir, texture_map):
        B, C, H, W = f_rgb.shape
        
        if texture_map.shape[-2:] != (H, W):
            texture_map = F.interpolate(
                texture_map, size=(H, W), mode='bilinear', align_corners=False
            )
        
        content_map = 1 - texture_map
        
        # 纹理引导增强（使用卷积而非注意力）
        f_rgb_texture = self.texture_guided_enhancement(f_rgb, texture_map)
        f_ir_texture = self.texture_guided_enhancement(f_ir, texture_map)
        
        # 内容引导增强
        f_rgb_content = self.content_enhance(f_rgb, f_ir, content_map)
        f_ir_content = self.content_enhance(f_ir, f_rgb, content_map)
        
        # 自适应融合
        f_rgb_fused = content_map * f_rgb_content + texture_map * f_rgb_texture
        f_ir_fused = content_map * f_ir_content + texture_map * f_ir_texture
        
        # 光照感知
        L = self.illumination_estimator(f_rgb_fused)
        
        f_rgb_out = f_rgb_fused + (1 - L) * self.interact_ir(f_ir_fused)
        f_ir_out = f_ir_fused + L * self.interact_rgb(f_rgb_fused)
        
        out_rgb = f_rgb + self.res_weight * self.norm_rgb(f_rgb_out)
        out_ir = f_ir + self.res_weight * self.norm_ir(f_ir_out)
        
        return out_rgb, out_ir
    
    def texture_guided_enhancement(self, f, texture_map):
        """
        纹理引导增强 - 使用卷积而非注意力
        借鉴TCDGNet思想:只在纹理区域增强,但用卷积实现
        """
        # 计算纹理掩码（区域级别的纹理强度）
        patch_size = max(4, min(f.shape[-2], f.shape[-1]) // 8)
        local_texture = F.avg_pool2d(texture_map, patch_size, stride=patch_size)
        threshold = local_texture.mean(dim=[2, 3], keepdim=True)
        texture_mask = (local_texture > threshold).float()
        texture_mask = F.interpolate(texture_mask, size=f.shape[-2:], mode='nearest')
        
        # 纹理区域增强
        f_masked = f * texture_mask
        
        # 卷积增强
        enhancement = self.texture_conv(f_masked)
        
        # 只增强纹理区域
        out = f + enhancement * texture_mask
        
        # FFN
        out = out + self.texture_ffn(out)
        
        return out
    
# ============================================================================
# 创新点1: 频空协同 (Frequency-Spatial Collaboration)
# ============================================================================

class SpectralCrossAttention(nn.Module):
    """
    谱域交叉注意力 (Spectral Cross-Attention, SCA)
    
    创新点：在频域中进行跨模态注意力计算
    - 利用幅度谱捕获全局结构相似性
    - 线性复杂度 O(Nd) 替代传统 O(N²)
    
    This module performs cross-modal attention in frequency domain,
    leveraging amplitude spectrum for global structural matching.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 频域投影层
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # 频域自适应权重 (Frequency-Adaptive Weight)
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_q: Query features from modality A [B, C, H, W]
            x_kv: Key-Value features from modality B [B, C, H, W]
        Returns:
            Enhanced features [B, C, H, W]
        """
        B, C, H, W = x_q.shape
        
        # 1. 投影到频域交互空间
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)
        
        # 2. 多头重排
        q = q.view(B, self.num_heads, self.head_dim, -1)  # [B, H, d, N]
        k = k.view(B, self.num_heads, self.head_dim, -1)  # [B, H, d, N]
        v = v.view(B, self.num_heads, self.head_dim, -1)  # [B, H, d, N]
        
        # 3. 线性注意力 (Performer风格核函数)
        # 使用ELU+1作为核函数，实现线性复杂度
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 4. 计算注意力 (线性复杂度 O(Nd²))
        # 先计算 K^T V
        kv = torch.matmul(k, v.transpose(-1, -2))  # [B, H, d, d]
        
        # 再计算 Q (K^T V)
        attn_out = torch.matmul(q.transpose(-1, -2), kv)  # [B, H, N, d]
        
        # 5. 归一化
        z = torch.matmul(q.transpose(-1, -2), k.sum(dim=-1, keepdim=True))
        attn_out = attn_out / (z + 1e-6)
        
        # 6. 恢复形状并应用频域权重
        out = attn_out.transpose(-1, -2).contiguous().view(B, C, H, W)
        out = out * self.freq_weight
        
        return self.out_proj(out)

from typing import Tuple, Optional, Dict
class AdaptiveFrequencyFusion(nn.Module):
    """
    自适应频域融合 (Adaptive Frequency Fusion, AFF)
    新增：
    1. 高频直通路径 (High-Frequency Pass-Through)
    2. 随机频域丢弃 (Spectral DropBlock)
    """
    def __init__(self, channels: int, high_freq_ratio: float = 0.05, drop_prob: float = 0.3):
        super().__init__()
        self.high_freq_ratio = high_freq_ratio
        self.drop_prob = drop_prob

        # 原有结构保持不变
        self.amp_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.pha_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.freq_mask = nn.Parameter(torch.ones(1, channels, 32, 16) * 0.5)
        self.spectral_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, fft_a: torch.Tensor, fft_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 提取幅度和相位
        amp_a, amp_b = torch.abs(fft_a), torch.abs(fft_b)
        pha_a, pha_b = torch.angle(fft_a), torch.angle(fft_b)

        # 2. 自适应幅度融合（原有）
        amp_cat = torch.cat([amp_a, amp_b], dim=1)
        amp_weight = torch.sigmoid(self.amp_fusion(amp_cat))
        amp_fused = amp_weight * amp_a + (1 - amp_weight) * amp_b

        # 3. 自适应相位融合（原有）
        pha_cat = torch.cat([pha_a, pha_b], dim=1)
        pha_weight = torch.sigmoid(self.pha_fusion(pha_cat))
        pha_fused = pha_weight * pha_a + (1 - pha_weight) * pha_b

        # 4. 频域掩码与频谱增强（原有）
        mask = F.interpolate(self.freq_mask, size=amp_fused.shape[-2:],
                            mode='bilinear', align_corners=False)
        amp_fused = amp_fused * torch.sigmoid(mask)
        amp_fused = self.spectral_enhance(amp_fused)

        # ==================== 新增改进 1：高频直通路径 ====================
        # 生成高频增强权重（高频区域 mask 值小，故 1-sigmoid(mask) 在高频区大）
        high_freq_weight = 1 - torch.sigmoid(mask)


        high_freq_cross = torch.abs(amp_a - amp_b)

        high_freq_cross = torch.tanh(high_freq_cross)

        # 将高频互相关特征按比例叠加到融合幅度上
        # amp_fused = amp_fused + self.high_freq_ratio * high_freq_weight * high_freq_cross
        amp_fused = amp_fused + 0.05 * high_freq_weight * high_freq_cross

        # ==================== 新增改进 2：随机频域丢弃 ====================
        if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
            # 随机丢弃部分通道：生成与 amp_fused 同形状的掩码，通道维度随机置零
            b, c, h, w = amp_fused.shape
            # 按通道随机丢弃，丢弃概率取 0.2
            channel_mask = (torch.rand(b, c, 1, 1, device=amp_fused.device) > 0.2).float()
            amp_fused = amp_fused * channel_mask

        return amp_fused, pha_fused


class SSCFv4(nn.Module):
    """频空协同模块 - 完整修复版"""
    def __init__(self, channels: int, reduction: int = 4, high_freq_ratio: float = 0.2, drop_prob: float = 0.3):
        super().__init__()
        hidden_dim = max(channels // reduction, 16)
        
        # 频域分支
        self.proj_freq_q = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_freq_k = nn.Conv2d(channels, hidden_dim, 1)
        self.sca_amp = SpectralCrossAttention(hidden_dim)
        self.sca_pha = SpectralCrossAttention(hidden_dim)
        self.aff = AdaptiveFrequencyFusion(hidden_dim,
                                           high_freq_ratio=high_freq_ratio,
                                           drop_prob=drop_prob)
        
        # 空间域分支
        self.proj_spatial_q = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_spatial_k = nn.Conv2d(channels, hidden_dim, 1)
        self.magic_cube = CrossModalMagicCube3D(channels)  # 使用原始通道数
        
        # 双向交互
        # self.freq_to_spatial = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim, 1),
        #     nn.Sigmoid()
        # )
        self.spatial_to_freq = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Tanh()
        )
        
        # 门控融合
        self.gate_freq = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self.gate_spatial = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # 输出
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.freq_beta = nn.Parameter(torch.tensor(0.3))
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

        
    def forward(self, x) -> torch.Tensor:

        f_rgb, f_ir = x[0], x[1]
        B, C, H, W = f_rgb.shape
        
        # 频域处理
        q_freq = self.proj_freq_q(f_rgb)
        k_freq = self.proj_freq_k(f_ir)
        
        fft_q = torch.fft.rfft2(q_freq, norm='ortho')
        fft_k = torch.fft.rfft2(k_freq, norm='ortho')
        
        amp_fused, pha_fused = self.aff(fft_q, fft_k)
        
        amp_q, amp_k = torch.abs(fft_q), torch.abs(fft_k)
        pha_q, pha_k = torch.angle(fft_q), torch.angle(fft_k)
        
        amp_enhanced = self.sca_amp(amp_q, amp_k)
        pha_enhanced = self.sca_pha(pha_q, pha_k)
        
        amp_final = amp_fused + 0.05 * amp_enhanced
        pha_final = pha_fused + 0.2 * pha_enhanced
        
        freq_features = torch.fft.irfft2(
            torch.complex(amp_final * torch.cos(pha_final),
                         amp_final * torch.sin(pha_final)),
            s=(H, W), norm='ortho'
        )
        
        # 空间域处理
        spatial_features = self.magic_cube(f_rgb, f_ir)
        
        # 投影空间特征到相同维度
        spatial_features_proj = self.proj_spatial_q(spatial_features)
        
        # 双向交互
        # 高频响应
        freq_energy = torch.mean(torch.abs(freq_features), dim=1, keepdim=True)

        # 归一化
        freq_energy = freq_energy / (
            freq_energy.amax(dim=(2,3), keepdim=True) + 1e-6
        )

        # 高频显著区域
        freq_guidance = 0.5 + 0.5 * freq_energy
        spatial_refined = spatial_features_proj * freq_guidance
        
        spatial_feedback = self.spatial_to_freq(spatial_features_proj)
        freq_refined = freq_features + 0.1 * self.freq_beta * spatial_feedback
        
        # 门控融合
        # 对两个分支分别做 LayerNorm
        freq_norm = F.layer_norm(freq_refined, freq_refined.shape[1:])
        spatial_norm = F.layer_norm(spatial_refined, spatial_refined.shape[1:])

        # 用归一化后的特征计算门控权重
        concat_norm = torch.cat([freq_norm, spatial_norm], dim=1)
        gate_logits = torch.cat([
            self.gate_freq(concat_norm),
            self.gate_spatial(concat_norm)
        ], dim=1)

        gate_weights = F.softmax(gate_logits, dim=1)

        gate_freq_weight = gate_weights[:, 0:1]
        gate_spatial_weight = gate_weights[:, 1:2]

        # 加权融合
        fused = gate_freq_weight * freq_norm + gate_spatial_weight * spatial_norm
                
        # 输出
        out = self.output_proj(
            torch.cat([
                fused,
                freq_refined + spatial_refined
            ], dim=1)
        )
        residual = self.alpha * f_rgb + (1 - self.alpha) * f_ir

        out = residual + self.residual_weight * out
        
        return out
# ============================================================================
# 创新点2: 三维跨模态魔方融合 (3D Cross-Modal Magic Cube)
# ============================================================================

class DynamicOffsetPredictor(nn.Module):
    """
    动态偏移预测器 - 修复BatchNorm在1x1特征图上的问题
    
    改进：
    1. 检测特征图尺寸，自适应选择归一化方式
    2. 使用全局平均池化后的特征预测偏移
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # 使用1x1卷积处理全局池化特征，避免BatchNorm问题
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 4, 1)  # 输出4个偏移值
        )
        
        # 偏移范围限制
        self.max_offset = 0.15
        self.register_buffer('offset_scale', torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = f_rgb.shape[0]
        H, W = f_rgb.shape[2], f_rgb.shape[3]
        
        # 全局平均池化获取全局上下文
        rgb_pool = F.adaptive_avg_pool2d(f_rgb, 1)  # [B, C, 1, 1]
        ir_pool = F.adaptive_avg_pool2d(f_ir, 1)    # [B, C, 1, 1]
        
        # 拼接并预测偏移
        concat = torch.cat([rgb_pool, ir_pool], dim=1)  # [B, 2C, 1, 1]
        offsets = self.offset_net(concat)  # [B, 4, 1, 1]
        
        # 分离RGB和IR的偏移并限制范围
        offsets = torch.tanh(offsets) * self.max_offset
        
        offset_rgb_h = offsets[:, 0:1]  # [B, 1, 1, 1]
        offset_rgb_w = offsets[:, 1:2]
        offset_ir_h = offsets[:, 2:3]
        offset_ir_w = offsets[:, 3:4]
        
        # 扩展到空间维度
        offset_rgb = torch.cat([
            offset_rgb_h.expand(-1, -1, H, W),
            offset_rgb_w.expand(-1, -1, H, W)
        ], dim=1)
        
        offset_ir = torch.cat([
            offset_ir_h.expand(-1, -1, H, W),
            offset_ir_w.expand(-1, -1, H, W)
        ], dim=1)
        
        return offset_rgb, offset_ir


class AxialAttention3D(nn.Module):
    """三维轴向注意力 - 修复版"""
    def __init__(self, channels: int, axis: str):
        super().__init__()
        self.axis = axis
        self.channels = channels
        
        if axis == 'H':
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, (1, 7), padding=(0, 3)),
                nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity()
            )
        elif axis == 'W':
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, (7, 1), padding=(3, 0)),
                nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity()
            )
        elif axis == 'C':
            # 通道轴使用全局池化
            self.conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(channels // 4, 8), 1),
                nn.GELU(),
                nn.Conv2d(max(channels // 4, 8), channels, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown axis: {axis}")
            
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == 'C':
            return self.conv(x)
        else:
            attn = self.conv(x)
            attn = torch.sigmoid(attn)
            return attn


class CrossModalMagicCube3D(nn.Module):
    """
    三维跨模态魔方模块 - 完整修复版
    
    修复内容：
    1. 所有BatchNorm替换为GroupNorm或条件判断
    2. 动态偏移预测适配小特征图
    3. 网格采样添加保护
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(channels // reduction, 16)  # 保证最小通道数
        
        # ========== 特征投影 ==========
        self.proj_rgb = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_ir = nn.Conv2d(channels, hidden_dim, 1)
        
        # ========== 动态偏移预测 ==========
        self.offset_predictor = DynamicOffsetPredictor(hidden_dim)
        
        # ========== 三轴注意力 ==========
        self.h_attn_rgb = AxialAttention3D(hidden_dim, 'H')
        self.h_attn_ir = AxialAttention3D(hidden_dim, 'H')
        self.w_attn_rgb = AxialAttention3D(hidden_dim, 'W')
        self.w_attn_ir = AxialAttention3D(hidden_dim, 'W')
        self.c_attn = AxialAttention3D(hidden_dim, 'C')
        
        # ========== 跨模态轴向交互 ==========
        self.cross_h = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.cross_w = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # ========== 立方体交互 ==========
        self.cubic_interaction = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )
        
        # ========== 跨轴信息传播 ==========
        self.axis_propagation = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        )
        
        # ========== 输出投影 ==========
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.beta = nn.Parameter(torch.ones(1) * 0.3)
        
    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor, 
                fused_spatial: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = f_rgb.shape
        
        # Step 1: 特征投影
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        
        # Step 2: 动态偏移补偿
        offset_rgb, offset_ir = self.offset_predictor(q_rgb, k_ir)
        
        # 应用偏移（使用grid_sample）
        try:
            theta_rgb = torch.eye(2, 3, device=f_rgb.device).unsqueeze(0).repeat(B, 1, 1)
            theta_ir = torch.eye(2, 3, device=f_rgb.device).unsqueeze(0).repeat(B, 1, 1)
            
            # 安全地计算平均偏移
            offset_rgb_h_mean = offset_rgb[:, 0].mean(dim=[1, 2], keepdim=True)
            offset_rgb_w_mean = offset_rgb[:, 1].mean(dim=[1, 2], keepdim=True)
            offset_ir_h_mean = offset_ir[:, 0].mean(dim=[1, 2], keepdim=True)
            offset_ir_w_mean = offset_ir[:, 1].mean(dim=[1, 2], keepdim=True)
            
            theta_rgb[:, 0, 2] = offset_rgb_w_mean.squeeze()
            theta_rgb[:, 1, 2] = offset_rgb_h_mean.squeeze()
            theta_ir[:, 0, 2] = offset_ir_w_mean.squeeze()
            theta_ir[:, 1, 2] = offset_ir_h_mean.squeeze()
            
            grid_rgb = F.affine_grid(theta_rgb, q_rgb.size(), align_corners=False)
            grid_ir = F.affine_grid(theta_ir, k_ir.size(), align_corners=False)
            
            q_rgb_aligned = F.grid_sample(q_rgb, grid_rgb, align_corners=False)
            k_ir_aligned = F.grid_sample(k_ir, grid_ir, align_corners=False)
        except Exception as e:
            # 如果grid_sample失败，直接使用原始特征
            print(f"Warning: grid_sample failed, using original features. Error: {e}")
            q_rgb_aligned = q_rgb
            k_ir_aligned = k_ir
        
        # Step 3: 三轴独立注意力
        h_attn_rgb = self.h_attn_rgb(q_rgb_aligned)
        h_attn_ir = self.h_attn_ir(k_ir_aligned)
        h_feat = h_attn_rgb * q_rgb_aligned + h_attn_ir * k_ir_aligned
        
        w_attn_rgb = self.w_attn_rgb(q_rgb_aligned)
        w_attn_ir = self.w_attn_ir(k_ir_aligned)
        w_feat = w_attn_rgb * q_rgb_aligned + w_attn_ir * k_ir_aligned
        
        # Step 4: 跨模态轴向交互
        # H轴
        h_feat_pool = h_feat.mean(dim=2, keepdim=True)
        k_ir_pool_h = k_ir_aligned.mean(dim=2, keepdim=True)
        h_cross_input = torch.cat([h_feat_pool, k_ir_pool_h], dim=1)
        h_cross_weight = self.cross_h(h_cross_input)
        h_feat_cross = h_feat * h_cross_weight
        
        # W轴
        w_feat_pool = w_feat.mean(dim=3, keepdim=True)
        k_ir_pool_w = k_ir_aligned.mean(dim=3, keepdim=True)
        w_cross_input = torch.cat([w_feat_pool, k_ir_pool_w], dim=1)
        w_cross_weight = self.cross_w(w_cross_input)
        w_feat_cross = w_feat * w_cross_weight
        
        # Step 5: C轴模态对齐
        if fused_spatial is not None:
            c_input = self.proj_rgb(fused_spatial)
        else:
            c_input = q_rgb_aligned + k_ir_aligned
        
        c_attn_weight = self.c_attn(c_input)
        c_feat = c_input * c_attn_weight
        
        # Step 6: 立方体交互
        cubic_input = torch.cat([h_feat_cross, w_feat_cross, c_feat], dim=1)
        cubic_feat = self.cubic_interaction(cubic_input)
        
        # Step 7: 跨轴传播
        magic_cube_out = self.axis_propagation(cubic_feat)
        
        # Step 8: 输出
        if fused_spatial is not None:
            base = fused_spatial
        else:
            base = f_rgb + f_ir
            
        out = self.output_proj(magic_cube_out)
        final_out = base + self.beta * out
        
        return final_out
    
    def get_attention_maps(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> dict:
        """
        可视化接口：返回三轴注意力图用于论文可视化
        
        Returns:
            Dict containing H, W, C attention maps
        """
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        
        return {
            'H_attention': self.h_attn_rgb(q_rgb),
            'W_attention': self.w_attn_rgb(q_rgb),
            'C_attention': self.c_attn(q_rgb)
        }

"""
FSCNet: Frequency-Spatial Collaborative Network with 3D Magic Cube
Paper: FSCNet for Cross-Modal Oriented Object Detection

Core Innovations:
1. Frequency-Spatial Collaboration (FSC)
2. Cross-Modal Magic Cube in 3D (CM-MC³)

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class ConditionalBatchNorm2d(nn.Module):
    """
    条件BatchNorm：当空间维度为1时自动跳过
    解决训练时BatchNorm需要batch_size>1或空间维度>1的问题
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.use_bn = True  # 训练时自动判断
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查空间维度是否足够进行BatchNorm
        if x.shape[-1] == 1 and x.shape[-2] == 1:
            # 空间维度为1x1时，使用InstanceNorm或直接跳过
            return x
        if x.shape[0] == 1 and self.training:
            # Batch size为1且训练时，使用InstanceNorm
            return F.instance_norm(x)
        return self.bn(x)


class SafeConvBlock(nn.Module):
    """安全的卷积块，自动处理小特征图"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.norm = ConditionalBatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    




##-------------------------------------------------
class ContentGuidedEnhancementv2(nn.Module):
    """内容引导增强 - 优化内存版本（保持原有逻辑）"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.use_attention = True
        self.channels = channels
        self.reduction = reduction
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        r = max(channels // reduction, 16)
        self.query = nn.Conv2d(channels, r, 1)
        self.key = nn.Conv2d(channels, r, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, f_main, f_aux, content_map):
        B, C, H, W = f_main.shape
        
        # 改进3：内容区域平滑去噪（使用平均池化低通滤波）
        if self.training and H * W > 16384:  # 仅训练时对高分辨率特征平滑
            f_main_smooth = F.avg_pool2d(f_main, 3, stride=1, padding=1)
            f_aux_smooth = F.avg_pool2d(f_aux, 3, stride=1, padding=1)
            # 内容区域使用平滑特征，纹理区域保持原样
            f_main = content_map * f_main_smooth + (1 - content_map) * f_main
            f_aux = content_map * f_aux_smooth + (1 - content_map) * f_aux
        
        # 内容加权
        f_main_weighted = f_main * content_map
        f_aux_weighted = f_aux * content_map
        
        if H * W > 16384:
            f_main_enhanced = self.depthwise_conv(f_main_weighted)
            f_aux_enhanced = self.depthwise_conv(f_aux_weighted)
            x = f_main_enhanced + f_aux_enhanced
        else:
            if H * W > 4096:
                scale = 2
                f_main_small = F.avg_pool2d(f_main_weighted, scale)
                f_aux_small = F.avg_pool2d(f_aux_weighted, scale)
                _, _, Hs, Ws = f_main_small.shape
                
                Q = self.query(f_main_small).view(B, -1, Hs*Ws)
                K = self.key(f_aux_small).view(B, -1, Hs*Ws)
                V = self.value(f_aux_small).view(B, C, Hs*Ws)
                
                attn = F.softmax(Q.transpose(-2, -1) @ K / (Q.shape[1] ** 0.5), dim=-1)
                x_small = (V @ attn).view(B, C, Hs, Ws)
                x = F.interpolate(x_small, size=(H, W), mode='bilinear', align_corners=False)
            else:
                Q = self.query(f_main_weighted).view(B, -1, H*W)
                K = self.key(f_aux_weighted).view(B, -1, H*W)
                V = self.value(f_aux_weighted).view(B, C, H*W)
                
                attn = F.softmax(Q.transpose(-2, -1) @ K / (Q.shape[1] ** 0.5), dim=-1)
                x = (V @ attn).view(B, C, H, W)
        
        out = self.fusion(torch.cat([f_main, self.gamma * x], dim=1))
        return out
    
class TIGEv6(nn.Module):
    """
    TIGEv5 改进版
    改进1：软纹理权重 + 边缘互补
    改进2：动态跨模态注入系数
    改进3：内容增强噪声抑制（集成在 ContentGuidedEnhancementv2 中）
    改进4：多尺度纹理图自适应融合
    """
    def __init__(self, channels, reduction=4, num_heads=4, window_size=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.reduction = reduction
        self.window_size = window_size
        
        # ========== 改进4：多尺度纹理图预处理核 ==========
        # 生成高斯核函数
        self.register_buffer('gaussian_5x5', self._get_gaussian_kernel(5, 0.8))
        self.register_buffer('gaussian_3x3', self._get_gaussian_kernel(3, 0.5))
        
        # ========== 1. 纹理引导卷积 ==========
        self.texture_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.texture_ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1)
        )
        
        # ========== 改进1：Sobel 算子用于边缘检测 ==========
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # ========== 2. 内容引导增强（含改进3噪声抑制）==========
        self.content_enhance = ContentGuidedEnhancementv2(channels, reduction)
        
        # ========== 3. 光照感知 ==========
        self.illumination_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.GELU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # ========== 改进2：动态跨模态注入系数网络 ==========
        self.dyn_gamma_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        self.dyn_gamma_ir = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # ========== 4. 跨模态交互 ==========
        self.interact_ir = nn.Conv2d(channels, channels, 1)
        self.interact_rgb = nn.Conv2d(channels, channels, 1)
        
        # ========== 5. 归一化 ==========
        self.norm_rgb = nn.BatchNorm2d(channels)
        self.norm_ir = nn.BatchNorm2d(channels)
        
        self.res_weight = nn.Parameter(torch.tensor(0.1))
    
    def _get_gaussian_kernel(self, size, sigma):
        """生成2D高斯核"""
        ax = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def adapt_texture_map(self, texture_map, target_size):
        """
        改进4：多尺度纹理图自适应融合
        - 根据目标特征图尺寸调整纹理图尺度
        - 大尺寸特征（P3）保留原始精细纹理
        - 小尺寸特征（P5）适度平滑，抑制高频噪声
        """
        H, W = target_size
        
        # 尺寸对齐
        if texture_map.shape[-2:] != (H, W):
            texture_map = F.interpolate(texture_map, size=(H, W), mode='bilinear', align_corners=False)
        
        # 根据特征图大小选择高斯平滑核
        if H <= 32:  # P5 层，强平滑
            kernel = self.gaussian_5x5
            texture_map = F.conv2d(texture_map, kernel, padding=2)
        elif H <= 64:  # P4 层，轻平滑
            kernel = self.gaussian_3x3
            texture_map = F.conv2d(texture_map, kernel, padding=1)
        # P3 层（H>64）不做平滑，保留原始细节
        
        return texture_map
    
    def texture_guided_enhancement(self, f, texture_map):
        """
        改进1：软纹理权重 + 边缘互补增强
        """
        B, C, H, W = f.shape
        
        # 1. 软纹理权重（连续映射，保留梯度）
        soft_texture_weight = torch.sigmoid((texture_map - 0.5) * 5.0)  # [B,1,H,W]
        
        # 2. 边缘权重（对输入特征计算梯度幅值）
        f_gray = f.mean(dim=1, keepdim=True)  # 转为单通道
        grad_x = torch.abs(F.conv2d(f_gray, self.sobel_x, padding=1))
        grad_y = torch.abs(F.conv2d(f_gray, self.sobel_y, padding=1))
        edge_strength = (grad_x + grad_y) / 2.0
        edge_weight = torch.sigmoid(edge_strength * 3.0)  # 放大边缘响应
        
        # 3. 纹理与边缘互补融合（并集逻辑）
        combined_weight = soft_texture_weight + edge_weight - soft_texture_weight * edge_weight
        
        # 4. 应用增强（仅对高权重区域）
        f_masked = f * combined_weight
        enhancement = self.texture_conv(f_masked)
        out = f + enhancement * combined_weight
        
        # 5. FFN 增强
        out = out + self.texture_ffn(out)
        
        return out
    
    def forward(self, f_rgb, f_ir, texture_map):
        B, C, H, W = f_rgb.shape
        
        # 改进4：多尺度纹理图自适应
        texture_map = self.adapt_texture_map(texture_map, (H, W))
        
        content_map = 1 - texture_map
        
        # 纹理引导增强（改进1已集成）
        f_rgb_texture = self.texture_guided_enhancement(f_rgb, texture_map)
        f_ir_texture = self.texture_guided_enhancement(f_ir, texture_map)
        
        # 内容引导增强（改进3已集成在模块内部）
        f_rgb_content = self.content_enhance(f_rgb, f_ir, content_map)
        f_ir_content = self.content_enhance(f_ir, f_rgb, content_map)
        
        # 自适应融合（纹理区域用纹理增强，内容区域用内容增强）
        f_rgb_fused = content_map * f_rgb_content + texture_map * f_rgb_texture
        f_ir_fused = content_map * f_ir_content + texture_map * f_ir_texture
        
        # 光照感知（全局光照系数）
        L = self.illumination_estimator(f_rgb_fused)
        
        # 改进2：动态跨模态注入系数
        gamma_rgb = self.dyn_gamma_rgb(self.interact_ir(f_ir_fused))
        gamma_ir = self.dyn_gamma_ir(self.interact_rgb(f_rgb_fused))
        
        # 非对称跨模态交互
        f_rgb_out = f_rgb_fused + gamma_rgb * (1 - L) * self.interact_ir(f_ir_fused)
        f_ir_out = f_ir_fused + gamma_ir * L * self.interact_rgb(f_rgb_fused)
        
        # 残差连接
        out_rgb = f_rgb + self.res_weight * self.norm_rgb(f_rgb_out)
        out_ir = f_ir + self.res_weight * self.norm_ir(f_ir_out)
        
        return out_rgb, out_ir

# ======================================================================
# 消融模块1：纯频域处理 (Frequency-Only)
# 直接复用 SSCFv5 的频域分支，去除空间交互与门控融合
# ======================================================================
class FreqOnlyModule(nn.Module):
    """
    消融实验：单一频域处理
    输入: (f_rgb, f_ir) 各为 [B, C, H, W]
    输出: [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 4,
                 high_freq_ratio: float = 0.05, drop_prob: float = 0.3):
        super().__init__()
        hidden_dim = max(channels // reduction, 16)

        # 频域分支（与原SSCFv5一致）
        self.proj_freq_q = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_freq_k = nn.Conv2d(channels, hidden_dim, 1)
        self.sca_amp = SpectralCrossAttention(hidden_dim)
        self.sca_pha = SpectralCrossAttention(hidden_dim)
        self.aff = AdaptiveFrequencyFusion(
            hidden_dim,
            high_freq_ratio=high_freq_ratio,
            drop_prob=drop_prob
        )
        self.band_low = 0.85
        self.band_high = 1.15

        # 频域输出投影（将 hidden_dim 映射回 channels）
        self.freq_out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def apply_bandpass(self, x):
        B, C, H, W = x.shape
        y, xv = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing='ij'
        )
        r = torch.sqrt(xv**2 + y**2)
        mask = torch.ones_like(r)
        mask[r < 0.2] = self.band_low
        mask[r > 0.6] = self.band_high
        return x * mask[None, None]

    def forward(self, x) -> torch.Tensor:
        f_rgb, f_ir = x[0], x[1]
        B, C, H, W = f_rgb.shape

        # ========== 频域处理 ==========
        q_freq = self.proj_freq_q(f_rgb)
        k_freq = self.proj_freq_k(f_ir)

        fft_q = torch.fft.rfft2(q_freq, norm='ortho')
        fft_k = torch.fft.rfft2(k_freq, norm='ortho')

        # 幅度/相位融合
        amp_fused, pha_fused = self.aff(fft_q, fft_k)

        # SCA增强
        amp_q, amp_k = torch.abs(fft_q), torch.abs(fft_k)
        pha_q, pha_k = torch.angle(fft_q), torch.angle(fft_k)

        amp_enhanced = self.sca_amp(amp_q, amp_k)
        pha_enhanced = self.sca_pha(pha_q, pha_k)

        amp_final = self.apply_bandpass(amp_fused + 0.1 * amp_enhanced)
        pha_final = pha_fused + 0.1 * pha_enhanced

        freq_features = torch.fft.irfft2(
            torch.complex(amp_final * torch.cos(pha_final),
                         amp_final * torch.sin(pha_final)),
            s=(H, W), norm='ortho'
        )

        # ========== 输出投影 + 残差 ==========
        out = self.freq_out_proj(freq_features)
        residual = self.alpha * f_rgb + (1 - self.alpha) * f_ir
        return out + residual

# ======================================================================
# 消融模块2：纯空域处理 (Spatial-Only)
# 直接复用 MagicCube 空间分支，去除频域分支与门控融合
# ======================================================================
class SpatialOnlyModule(nn.Module):
    """
    消融实验：单一空域处理
    输入: (f_rgb, f_ir) 各为 [B, C, H, W]
    输出: [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(channels // reduction, 16)

        # 空间分支（与原SSCFv5一致）
        self.magic_cube = CrossModalMagicCube3D(channels)
        self.proj_spatial = nn.Conv2d(channels, hidden_dim, 1)

        # 空间输出投影（将 hidden_dim 映射回 channels）
        self.spatial_out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x) -> torch.Tensor:
        f_rgb, f_ir = x[0], x[1]

        # ========== 空间域处理 ==========
        spatial_features = self.magic_cube(f_rgb, f_ir)        # [B, C, H, W]
        spatial_proj = self.proj_spatial(spatial_features)     # [B, hidden_dim, H, W]

        # ========== 输出投影 + 残差 ==========
        out = self.spatial_out_proj(spatial_proj)
        residual = self.alpha * f_rgb + (1 - self.alpha) * f_ir
        return out + residual

# ============================================================================
# 创新点1: 频空协同 (Frequency-Spatial Collaboration)
# ============================================================================
class SpectralCrossAttention(nn.Module):
    """
    谱域交叉注意力 (Spectral Cross-Attention, SCA)
    整合 v5/v6 改进：可学习频域位置编码 + 局部响应归一化
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # v5/v6 新增：局部对比度归一化，增强高频响应
        self.local_norm = nn.LocalResponseNorm(size=5)
        # v5/v6 新增：可学习频域位置编码（预设尺寸 64x33，插值适配）
        self.freq_pos_enc = nn.Parameter(torch.randn(1, channels, 64, 33) * 0.02)
        
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_q.shape

        # 局部响应归一化
        x_q = self.local_norm(x_q)
        x_kv = self.local_norm(x_kv)

        # 注入频域位置编码
        pos_enc = F.interpolate(self.freq_pos_enc, size=(H, W),
                                mode='bilinear', align_corners=False)
        x_q = x_q + pos_enc
        x_kv = x_kv + pos_enc
        
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)
        
        q = q.view(B, self.num_heads, self.head_dim, -1)
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)
        
        # 线性注意力
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        kv = torch.matmul(k, v.transpose(-1, -2))
        attn_out = torch.matmul(q.transpose(-1, -2), kv)
        
        z = torch.matmul(q.transpose(-1, -2), k.sum(dim=-1, keepdim=True))
        attn_out = attn_out / (z + 1e-6)
        
        out = attn_out.transpose(-1, -2).contiguous().view(B, C, H, W)
        out = out * self.freq_weight
        return self.out_proj(out)


class AdaptiveFrequencyFusion(nn.Module):
    """
    自适应频域融合 (Final)
    整合 v6 混合环形相位 + 新版本幅度差高频直通
    """
    def __init__(self, channels: int, high_freq_ratio: float = 0.05, drop_prob: float = 0.3):
        super().__init__()
        self.high_freq_ratio = high_freq_ratio
        self.drop_prob = drop_prob

        self.amp_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.pha_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.freq_mask = nn.Parameter(torch.ones(1, channels, 32, 16) * 0.5)
        self.spectral_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1)
        )

    def build_high_mask(self, H, W, device):
        """生成高频区域掩码 (v6)"""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        r = torch.sqrt(x**2 + y**2)
        high = (r > 0.6).float()
        return high[None, None]

    def forward(self, fft_a: torch.Tensor, fft_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        amp_a, amp_b = torch.abs(fft_a), torch.abs(fft_b)
        pha_a, pha_b = torch.angle(fft_a), torch.angle(fft_b)
        B, C, H, W = amp_a.shape

        # ---------- 幅度融合 ----------
        amp_cat = torch.cat([amp_a, amp_b], dim=1)
        amp_weight = torch.sigmoid(self.amp_fusion(amp_cat))
        amp_fused = amp_weight * amp_a + (1 - amp_weight) * amp_b

        mask = F.interpolate(self.freq_mask, size=(H, W),
                             mode='bilinear', align_corners=False)
        amp_fused = amp_fused * torch.sigmoid(mask)
        amp_fused = self.spectral_enhance(amp_fused)

        # 高频直通路径：使用幅度差 + tanh（新版本思路）
        high_freq_weight = 1 - torch.sigmoid(mask)
        high_freq_cross = torch.tanh(torch.abs(amp_a - amp_b))
        amp_fused = amp_fused + self.high_freq_ratio * high_freq_weight * high_freq_cross

        # ---------- 相位融合（v6 混合环形相位） ----------
        pha_cat = torch.cat([pha_a, pha_b], dim=1)
        pha_weight = torch.sigmoid(self.pha_fusion(pha_cat))

        # 正常加权相位
        normal_phase = pha_weight * pha_a + (1 - pha_weight) * pha_b
        # 环形平均相位（处理相位卷绕）
        circular_phase = torch.atan2(
            pha_weight * torch.sin(pha_a) + (1 - pha_weight) * torch.sin(pha_b),
            pha_weight * torch.cos(pha_a) + (1 - pha_weight) * torch.cos(pha_b)
        )

        # 仅在高频区域使用环形相位
        high_mask = self.build_high_mask(H, W, amp_a.device)
        pha_fused = normal_phase * (1 - high_mask) + circular_phase * high_mask

        # ---------- 随机频域丢弃 ----------
        if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
            channel_mask = (torch.rand(B, C, 1, 1, device=amp_fused.device) > 0.2).float()
            amp_fused = amp_fused * channel_mask

        return amp_fused, pha_fused

    

class FSCFv7(nn.Module):
    """
    最终版频空协同模块
    整合：v6 带通滤波 + 新版本门控融合与输出 + v5/v6 魔方
    """
    def __init__(self, channels: int, reduction: int = 4,
                 high_freq_ratio: float = 0.05, drop_prob: float = 0.3):
        super().__init__()
        hidden_dim = max(channels // reduction, 16)
        
        # 频域分支
        self.proj_freq_q = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_freq_k = nn.Conv2d(channels, hidden_dim, 1)
        self.sca_amp = SpectralCrossAttention(hidden_dim)
        self.sca_pha = SpectralCrossAttention(hidden_dim)
        self.aff = AdaptiveFrequencyFusion(hidden_dim,
                                           high_freq_ratio=high_freq_ratio,
                                           drop_prob=drop_prob)
        # 带通滤波（v6）
        self.band_low = 0.85
        self.band_high = 1.15
        
        # 空间域分支（v5/v6 条带卷积魔方）
        self.proj_spatial_q = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_spatial_k = nn.Conv2d(channels, hidden_dim, 1)
        self.magic_cube = CrossModalMagicCube3D_Final(channels)
        
        # 双向交互：可学习频→空引导（v4/v5/v6）
        self.freq_to_spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_to_freq = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Tanh()
        )
        # 空间反馈可调节强度（新版本）
        self.freq_beta = nn.Parameter(torch.tensor(0.3))
        
        # 门控融合（新版本 LayerNorm + Softmax 竞争）
        self.gate_freq = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )
        self.gate_spatial = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )
        
        # 输出：双流拼接（新版本）
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
    def apply_bandpass(self, x: torch.Tensor) -> torch.Tensor:
        """v6 带通滤波：抑制低频、增强高频"""
        B, C, H, W = x.shape
        y, xv = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing='ij'
        )
        r = torch.sqrt(xv**2 + y**2)
        mask = torch.ones_like(r)
        mask[r < 0.2] = self.band_low
        mask[r > 0.6] = self.band_high
        return x * mask[None, None]
        
    def forward(self, x) -> torch.Tensor:
        f_rgb, f_ir = x[0], x[1]
        B, C, H, W = f_rgb.shape
        
        # ---------- 频域处理 ----------
        q_freq = self.proj_freq_q(f_rgb)
        k_freq = self.proj_freq_k(f_ir)
        
        fft_q = torch.fft.rfft2(q_freq, norm='ortho')
        fft_k = torch.fft.rfft2(k_freq, norm='ortho')
        
        amp_fused, pha_fused = self.aff(fft_q, fft_k)
        
        amp_q, amp_k = torch.abs(fft_q), torch.abs(fft_k)
        pha_q, pha_k = torch.angle(fft_q), torch.angle(fft_k)
        
        amp_enhanced = self.sca_amp(amp_q, amp_k)
        pha_enhanced = self.sca_pha(pha_q, pha_k)
        
        # 重相位增强（0.2），轻幅度增强（0.05），再经带通滤波
        amp_final = self.apply_bandpass(amp_fused + 0.05 * amp_enhanced)
        pha_final = pha_fused + 0.2 * pha_enhanced
        
        freq_features = torch.fft.irfft2(
            torch.complex(amp_final * torch.cos(pha_final),
                         amp_final * torch.sin(pha_final)),
            s=(H, W), norm='ortho'
        )
        
        # ---------- 空间域处理 ----------
        spatial_features = self.magic_cube(f_rgb, f_ir)
        spatial_features_proj = self.proj_spatial_q(spatial_features)
        
        # ---------- 双向交互 ----------
        freq_guidance = self.freq_to_spatial(freq_features)  # 可学习频→空
        spatial_refined = spatial_features_proj * freq_guidance
        
        spatial_feedback = self.spatial_to_freq(spatial_features_proj)
        freq_refined = freq_features + 0.1 * self.freq_beta * spatial_feedback
        
        # ---------- 门控融合（LayerNorm + Softmax 竞争） ----------
        freq_norm = F.layer_norm(freq_refined, freq_refined.shape[1:])
        spatial_norm = F.layer_norm(spatial_refined, spatial_refined.shape[1:])
        
        concat_norm = torch.cat([freq_norm, spatial_norm], dim=1)
        gate_logits = torch.cat([
            self.gate_freq(concat_norm),
            self.gate_spatial(concat_norm)
        ], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        gate_freq_weight = gate_weights[:, 0:1]
        gate_spatial_weight = gate_weights[:, 1:2]
        
        fused = gate_freq_weight * freq_norm + gate_spatial_weight * spatial_norm
        
        # ---------- 输出：双流拼接 + 可学习残差 ----------
        out = self.output_proj(
            torch.cat([fused, freq_refined + spatial_refined], dim=1)
        )
        residual = self.alpha * f_rgb + (1 - self.alpha) * f_ir
        out = residual + self.residual_weight * out
        
        return out
# ============================================================================
# 创新点2: 三维跨模态魔方融合 (3D Cross-Modal Magic Cube)
# ============================================================================

class DynamicOffsetPredictor(nn.Module):
    """
    动态偏移预测器 - 修复BatchNorm在1x1特征图上的问题
    
    改进：
    1. 检测特征图尺寸，自适应选择归一化方式
    2. 使用全局平均池化后的特征预测偏移
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # 使用1x1卷积处理全局池化特征，避免BatchNorm问题
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 4, 1)  # 输出4个偏移值
        )
        
        # 偏移范围限制
        self.max_offset = 0.15
        self.register_buffer('offset_scale', torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = f_rgb.shape[0]
        H, W = f_rgb.shape[2], f_rgb.shape[3]
        
        # 全局平均池化获取全局上下文
        rgb_pool = F.adaptive_avg_pool2d(f_rgb, 1)  # [B, C, 1, 1]
        ir_pool = F.adaptive_avg_pool2d(f_ir, 1)    # [B, C, 1, 1]
        
        # 拼接并预测偏移
        concat = torch.cat([rgb_pool, ir_pool], dim=1)  # [B, 2C, 1, 1]
        offsets = self.offset_net(concat)  # [B, 4, 1, 1]
        
        # 分离RGB和IR的偏移并限制范围
        offsets = torch.tanh(offsets) * self.max_offset
        
        offset_rgb_h = offsets[:, 0:1]  # [B, 1, 1, 1]
        offset_rgb_w = offsets[:, 1:2]
        offset_ir_h = offsets[:, 2:3]
        offset_ir_w = offsets[:, 3:4]
        
        # 扩展到空间维度
        offset_rgb = torch.cat([
            offset_rgb_h.expand(-1, -1, H, W),
            offset_rgb_w.expand(-1, -1, H, W)
        ], dim=1)
        
        offset_ir = torch.cat([
            offset_ir_h.expand(-1, -1, H, W),
            offset_ir_w.expand(-1, -1, H, W)
        ], dim=1)
        
        return offset_rgb, offset_ir


class AxialAttention3D(nn.Module):
    """三维轴向注意力 - 修复版"""
    def __init__(self, channels: int, axis: str):
        super().__init__()
        self.axis = axis
        self.channels = channels
        
        if axis == 'H':
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, (1, 7), padding=(0, 3)),
                nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity()
            )
        elif axis == 'W':
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, (7, 1), padding=(3, 0)),
                nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity()
            )
        elif axis == 'C':
            # 通道轴使用全局池化
            self.conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(channels // 4, 8), 1),
                nn.GELU(),
                nn.Conv2d(max(channels // 4, 8), channels, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown axis: {axis}")
            
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == 'C':
            return self.conv(x)
        else:
            attn = self.conv(x)
            attn = torch.sigmoid(attn)
            return attn


class CrossModalMagicCube3D_Final(nn.Module):
    """魔方模块最终版：条带卷积跨模态交互"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(channels // reduction, 16)
        
        self.proj_rgb = nn.Conv2d(channels, hidden_dim, 1)
        self.proj_ir = nn.Conv2d(channels, hidden_dim, 1)
        self.offset_predictor = DynamicOffsetPredictor(hidden_dim)
        
        self.h_attn_rgb = AxialAttention3D(hidden_dim, 'H')
        self.h_attn_ir = AxialAttention3D(hidden_dim, 'H')
        self.w_attn_rgb = AxialAttention3D(hidden_dim, 'W')
        self.w_attn_ir = AxialAttention3D(hidden_dim, 'W')
        self.c_attn = AxialAttention3D(hidden_dim, 'C')

        # 条带卷积跨模态交互（v5/v6）
        self.cross_h = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, (1, 7), padding=(0, 3)),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        self.cross_w = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, (7, 1), padding=(3, 0)),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.cubic_interaction = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim) if hidden_dim >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )
        self.axis_propagation = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, 1),
            nn.GroupNorm(min(8, channels), channels) if channels >= 8 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.beta = nn.Parameter(torch.ones(1) * 0.3)
        
    def forward(self, f_rgb, f_ir, fused_spatial=None):
        B, C, H, W = f_rgb.shape
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        
        offset_rgb, offset_ir = self.offset_predictor(q_rgb, k_ir)
        try:
            theta = torch.eye(2, 3, device=f_rgb.device).unsqueeze(0).repeat(B, 1, 1)
            theta_rgb = theta.clone()
            theta_ir = theta.clone()
            theta_rgb[:, 0, 2] = offset_rgb[:, 1].mean(dim=[1,2])
            theta_rgb[:, 1, 2] = offset_rgb[:, 0].mean(dim=[1,2])
            theta_ir[:, 0, 2] = offset_ir[:, 1].mean(dim=[1,2])
            theta_ir[:, 1, 2] = offset_ir[:, 0].mean(dim=[1,2])
            
            grid_rgb = F.affine_grid(theta_rgb, q_rgb.size(), align_corners=False)
            grid_ir = F.affine_grid(theta_ir, k_ir.size(), align_corners=False)
            q_rgb_aligned = F.grid_sample(q_rgb, grid_rgb, align_corners=False)
            k_ir_aligned = F.grid_sample(k_ir, grid_ir, align_corners=False)
        except:
            q_rgb_aligned, k_ir_aligned = q_rgb, k_ir
        
        h_attn_rgb = self.h_attn_rgb(q_rgb_aligned)
        h_attn_ir = self.h_attn_ir(k_ir_aligned)
        h_feat = h_attn_rgb * q_rgb_aligned + h_attn_ir * k_ir_aligned
        
        w_attn_rgb = self.w_attn_rgb(q_rgb_aligned)
        w_attn_ir = self.w_attn_ir(k_ir_aligned)
        w_feat = w_attn_rgb * q_rgb_aligned + w_attn_ir * k_ir_aligned
        
        # 全分辨率跨模态交互
        h_cross_input = torch.cat([h_feat, k_ir_aligned], dim=1)
        h_cross_weight = self.cross_h(h_cross_input)
        h_feat_cross = h_feat * h_cross_weight
        
        w_cross_input = torch.cat([w_feat, k_ir_aligned], dim=1)
        w_cross_weight = self.cross_w(w_cross_input)
        w_feat_cross = w_feat * w_cross_weight
        
        c_input = self.proj_rgb(fused_spatial) if fused_spatial is not None else q_rgb_aligned + k_ir_aligned
        c_attn_weight = self.c_attn(c_input)
        c_feat = c_input * c_attn_weight
        
        cubic_input = torch.cat([h_feat_cross, w_feat_cross, c_feat], dim=1)
        cubic_feat = self.cubic_interaction(cubic_input)
        magic_cube_out = self.axis_propagation(cubic_feat)
        
        base = fused_spatial if fused_spatial is not None else f_rgb + f_ir
        out = self.output_proj(magic_cube_out)
        return base + self.beta * out
    
    def get_attention_maps(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> dict:
        """
        可视化接口：返回三轴注意力图用于论文可视化
        
        Returns:
            Dict containing H, W, C attention maps
        """
        q_rgb = self.proj_rgb(f_rgb)
        k_ir = self.proj_ir(f_ir)
        
        return {
            'H_attention': self.h_attn_rgb(q_rgb),
            'W_attention': self.w_attn_rgb(q_rgb),
            'C_attention': self.c_attn(q_rgb)
        }
    
    

class CAFusion(TIGEv6):
    def __init__(self, channels, reduction_ratio=4): 
        super().__init__(channels, reduction_ratio)


class ADD(FSCFv7):
    def __init__(self, channels, reduction=4):
        super().__init__(channels, reduction)


# class CAFusion(CAFusion_base):
#     def __init__(self, channels, reduction_ratio=4):
#         super().__init__(channels, reduction_ratio)

# class ADD(base_ADD):
#     def __init__(self, channels):
#         super().__init__(channels)


