import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import torchvision.transforms as transforms

import cv2
# import glob
import os


# from lib.EfficientNet import EfficientNet
from model.lib.EfficientNet import EfficientNet

np.set_printoptions(suppress=True, threshold=1e5)

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

class BasicConv2dReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
#*****************************************************************************
class EEU(nn.Module):
    def __init__(self, in_channel):
        super(EEU, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.max_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)
        
        self.weight_EEU = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) 
        
        
        self.conv_edge_1 = nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(3, 1), padding=(1, 0)),
        )
        
        self.conv_edge_2 = nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(3, 1), padding=(1, 0)),
        )
        
        self.conv_edge_add = nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(3, 1), padding=(1, 0)),
        )

    def forward(self, x):
    
    
        # The code will be published later
        
        
        return self.PReLU(edge), out

class DEU(nn.Module):
    def __init__(self, in_channel):
        super(DEU, self).__init__()
        
        # self.kernel_size = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True) 
        # self.kernel_size = F.softmax(self.kernel_size, dim=0)
        
        self.blur_transform_0 = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
        self.blur_transform_1 = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
        self.blur_transform_2 = transforms.GaussianBlur(kernel_size=7, sigma=2.0)
        self.blur_transform_3 = transforms.GaussianBlur(kernel_size=9, sigma=4.0)
        
        self.weight_DEU = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True) 
        
        self.conv_2 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)
        
        self.conv_detail_0 = nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(3, 1), padding=(1, 0)),
        )
        
        self.conv_detail_1 = nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(5, 1), padding=(2, 0)),
        )
        
        self.conv_detail_2= nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(7, 1), padding=(3, 0)),
        )
        
        self.conv_detail_3= nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(9, 1), padding=(4, 0)),
        )
        
        self.conv_detail_add = nn.Sequential(
            BasicConv2dReLu(in_channel, in_channel, 1),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2dReLu(in_channel, in_channel, kernel_size=(3, 1), padding=(1, 0)),
        )

    def forward(self, x):
    
    
        # The code will be published later
        
        
        return self.PReLU(detail), out

class EDM(nn.Module):
    def __init__(self, channel1=1, channel2=1):
        super(EDM, self).__init__()
        
        self.eeu1 = EEU(channel1)
        self.deu1 = DEU(channel1)
        
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.eeu2 = EEU(channel2)
        self.deu2 = DEU(channel2)
        
        self.SA_1_e = SpatialAttention()
        self.SA_1_d = SpatialAttention()
        self.SA_2_e = SpatialAttention()
        self.SA_2_d = SpatialAttention()
        
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)        
        self.sa_fusion_1 = nn.Sequential(BasicConv2d(1, 1, kernel_size=(1, 3), padding=(0, 1)),
                                         BasicConv2d(1, 1, kernel_size=(3, 1), padding=(1, 0)),
                                         nn.Sigmoid()
                                         )
        self.sa_fusion_2 = nn.Sequential(BasicConv2d(1, 1, kernel_size=(1, 3), padding=(0, 1)),
                                         BasicConv2d(1, 1, kernel_size=(3, 1), padding=(1, 0)),
                                         nn.Sigmoid()
                                         )
 
    # torch.Size([1, 24, 112, 112])
    # torch.Size([1, 40, 56, 56])
    def forward(self, x1, x2):       

    
    
        # The code will be published later
        
        

        return x_1_out, x_2_out


#*****************************************************************************
    
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
    
# Hierarchical feature fusion
class HFF_high(nn.Module):
    def __init__(self, channel):
        super(HFF_high, self).__init__()
        
        self.SA_3 = SpatialAttention()
        self.SA_4 = SpatialAttention()
        self.SA_5 = SpatialAttention()
        self.CA = ChannelAttention(channel)

        self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_5 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_high = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x3, x4, x5):
        
    
    
        # The code will be published later
        
        

        return x_3_high, x_4_high, x_5_high
    
# def avg_pool2d(matrix, kernel_size, stride, padding=0):
#     # 计算填充后的矩阵尺寸
#     padded_height = matrix.shape[0] + 2 * padding
#     padded_width = matrix.shape[1] + 2 * padding
#     padded_matrix = np.pad(matrix, pad_width=((0, 0), (padding, padding)), mode='constant', constant_values=0)
    
#     # 计算输出矩阵的尺寸
#     output_height = (padded_height - kernel_size) // stride + 1
#     output_width = (padded_width - kernel_size) // stride + 1
    
#     # 初始化输出矩阵
#     pooled_matrix = np.zeros((output_height, output_width))
    
#     # 应用池化
#     for i in range(0, output_height):
#         for j in range(0, output_width):
#             # 计算窗口的起始和结束索引
#             start_i = i * stride
#             start_j = j * stride
#             end_i = start_i + kernel_size
#             end_j = start_j + kernel_size
            
#             matrix_detached = matrix.detach()
            
#             # 计算窗口内的平均值
#             pooled_matrix[i, j] = np.mean(matrix_detached[start_i:end_i, start_j:end_j])
    
#     return pooled_matrix

class att(nn.Module):
    def __init__(self, in_dim, topk):
        super(att, self).__init__()
        
        self.topk = topk
        self.query_conv_x = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.key_conv_x = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.value_conv_x = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv_xx = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv_xy = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        
        self.conv_xx_pool = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv_xy_pool = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
    
        self.query_conv_y = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.key_conv_y = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.value_conv_y = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv_yy = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv_yx = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        
        self.conv_yy_pool = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv_yx_pool = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        
        for layer in [self.query_conv_x, self.key_conv_x, self.value_conv_x, self.conv_xx, self.conv_xy, self.query_conv_y, self.key_conv_y, self.value_conv_y, self.conv_yy, self.conv_yx]:
            weight_init.c2_msra_fill(layer)
        
        self.scale = 1.0 / (in_dim ** 0.5)
        
        self.conv_cat = nn.Sequential(
            BasicConv2dReLu(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            BasicConv2dReLu(in_dim, in_dim, kernel_size=3, stride=1, padding=1))
        
        self.conv_cat_pool = nn.Sequential(
            BasicConv2dReLu(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            BasicConv2dReLu(in_dim, in_dim, kernel_size=3, stride=1, padding=1))
        
        self.conv_cat_final = nn.Sequential(
            BasicConv2dReLu(in_dim * 2, in_dim, kernel_size=1, stride=1, padding=0),
            BasicConv2dReLu(in_dim, in_dim, kernel_size=3, stride=1, padding=1))
        
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)

    
    def forward(self, x, y):
            
    
        # The code will be published later
        
        
    
        return cat_final

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # decode
        self.out_layer_5 = nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
        )
        
        self.out_layer_4 = nn.Sequential(
        nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
        )
            
        self.out_layer_3 = nn.Sequential(
        nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
        )
        
        self.out_layer_2 = nn.Sequential(
        nn.Conv2d(232, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
        )
        
        self.out_layer_1 = nn.Sequential(
        nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
        )

    def forward(self, r1, r2, r3, r4, r5):
        
    
    
        # The code will be published later
        
        
    
        return S_1_pred, S_2_pred, S_3_pred, S_4_pred, S_5_pred


class Network(nn.Module):
    # efficientnet-b5 based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
    # def __init__(self, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.context_encoder = EfficientNet.from_pretrained('efficientnet-b5')
        
        self.EDM = EDM()
        
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.rfb3 = RFB_modified(64, channel)
        self.rfb4 = RFB_modified(176, channel)
        self.rfb5 = RFB_modified(512, channel)
        self.HFF = HFF_high(channel)
            
        self.att_3 = att(channel, topk = 16)
        self.att_4 = att(channel, topk = 4)
        self.att_5 = att(channel, topk = 1)
        
        # self.att = att(channel)
        
        self.decoder = Decoder()

    def forward(self, x):
        # backbone
        endpoints = self.context_encoder.extract_endpoints(x)
        r1 = endpoints['reduction_1']
        r2 = endpoints['reduction_2']
        r3 = endpoints['reduction_3']
        r4 = endpoints['reduction_4']
        r5 = endpoints['reduction_5']
        
        # torch.Size([1, 24, 112, 112])
        # torch.Size([1, 40, 56, 56])
        # torch.Size([1, 64, 28, 28])
        # torch.Size([1, 176, 14, 14])
        # torch.Size([1, 512, 7, 7])
        
        # print(r1.shape)
        # print(r2.shape)
        # print(r3.shape)
        # print(r4.shape)
        # print(r5.shape)
        
        r3_1 = self.rfb3(r3)
        r4_1 = self.rfb4(r4)
        r5_1 = self.rfb5(r5)
        
        # torch.Size([1, 64, 28, 28])
        # torch.Size([1, 64, 14, 14])
        # torch.Size([1, 64, 7, 7])
        
        # print(r1.shape)
        # print(r2.shape)
        # print(r3_1.shape)
        # print(r4_1.shape)
        # print(r5_1.shape)
        
        x_1_out, x_2_out = self.EDM(r1, r2)
        # print(x_1_out.shape)
        # print(x_2_out.shape)
        
        
        
        x_3_high, x_4_high, x_5_high = self.HFF(r3_1, r4_1, r5_1)
           
        
        # print(x_3_high.shape)
        # print(x_4_high.shape)
        # print(x_5_high.shape)
        
        cat_3 = self.att_3(r3_1, x_3_high)
        cat_4 = self.att_4(r4_1, x_4_high)
        cat_5 = self.att_5(r5_1, x_5_high)
        # print(cat_3.shape)
        # print(cat_4.shape)
        # print(cat_5.shape)
        
        S_1_pred, S_2_pred, S_3_pred, S_4_pred, S_5_pred = self.decoder(x_1_out, x_2_out, cat_3, cat_4, cat_5)
        return S_1_pred, S_2_pred, S_3_pred, S_4_pred, S_5_pred

        

if __name__ == '__main__':
    import numpy as np
    from time import time
    from thop import profile
    
    from ptflops import get_model_complexity_info

    net = Network()
    net.eval()
    
    dump_x = torch.randn(1, 3, 224, 224)
    
    y = net(dump_x)
    
    flops, params = profile(net, (dump_x, ))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

    # #     #   The second method
    # flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)
    # # print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
    
    
    # frame_rate = np.zeros((1000, 1))
    # for i in range(1000):
    #     start = time()
    #     y = net(dump_x)
    #     end = time()
    #     running_frame_rate = 1 * float(1 / (end - start))
    #     print(i, '->', running_frame_rate)
    #     frame_rate[i] = running_frame_rate
    # print(np.mean(frame_rate))
    # print(y.shape)