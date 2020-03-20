# -*- coding: utf-8 -*-
'''
yolov3网络结构实现
'''

import torch

'''
上采样模块

为什么不用反卷积？
反卷积是网络层，需要训练；且反卷积比这个慢多了
'''
class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):

        #这里用的邻近插值，也可以用其它方式，改mode就行了；上采样为2倍，scale_factor=2
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


'''
直接定义每一个卷积层
'''
class ConvolutionalLayer(torch.nn.Module):

    #bias=False即默认偏置为False
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            #加批归一化是常规操作，能加速特征提取的速度
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


'''
定义残差单元，就是网络结构中的：
1*1
3*3
Residual

网络结构中使用了多个残差结构，每个残差单元仅是输入输出通道不同
'''

class ResidualLayer(torch.nn.Module):

    #传入in_channels，是因为每个残差单元输入和输出的通道数相同，
    # 中间是使用1*1来降通道，后面也使用了3*3来升了通道
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            #1*1的卷积，通道降为1半
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            #3*3的卷积，通道升回去；为了保证特征图大小不变，所以padding=1
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):

        #x是输入数据,self.sub_module(x)是这个残差单元的输出数据，
        #其实两个加起来就是对下一部分加了残差(也就是第一次的残差单元的输入是没加残差的)
        #其实也可以自己手动写上，只是这个代码没写上
        return x + self.sub_module(x)


class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)

'''
网络结构中的ConvolutionalSet模块，
主要作用就是加深网络
'''
class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)

'''
yolov3的整体网络(主网络+检测网络)
'''
class MainNet(torch.nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()

        #由于后面要做路由，所以这里trunk_52、trunk_26和trunk_13分布就是主网络中各个尺度的输出
        self.trunk_52 = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            ConvolutionalLayer(32, 64, 3, 2, 1),

            ResidualLayer(64),
            #下采样
            DownsamplingLayer(64, 128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )

        self.trunk_26 = torch.nn.Sequential(
            DownsamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )

        self.trunk_13 = torch.nn.Sequential(
            DownsamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        #trunk_13就是主网络的最后一部分(4x的部分)，要传入ConvolutionalSet，随后做13尺度的检测和上采样
        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        #13尺度的检测
        self.detetion_13 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            #注意输出45是10个类别的输出，即(置信度+中心点(2个值)+宽高(2个值)+10分类)*3
            #乘3是因为每个尺度都有3种建议框
            torch.nn.Conv2d(1024, 45, 1, 1, 0)
        )

        #上采样为26，输入就是convset_13的输出
        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpsampleLayer()
        )

        #传入ConvolutionalSet，随后做26尺度的检测和上采样
        #输入为768,是因为路由是叠加通道，下面的通道数为256，上面的为512，加起来就是768
        #下面的26*26是有13*13上采样过来的，上采样的过程中会丢失信息，即下面的26*26实则比上面的26*26包含的信息少，
        # 因此使下面的通道少于上面的通道(下面的是256，上面的是512)，使网络参考的信息更加多的参考原始数据，即提取特征时更加侧重原始的信息。
        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(768, 256)
        )

        #26尺度的检测
        self.detetion_26 = torch.nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),

            #三个尺度输出通道数要一致
            torch.nn.Conv2d(512, 45, 1, 1, 0)
        )

        # 上采样为52，输入就是convset_26的输出
        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpsampleLayer()
        )

        #传入ConvolutionalSet，随后做52尺度的检测
        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(384, 128)
        )

        #52尺度的检测
        self.detetion_52 = torch.nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 45, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        #torch.cat就是做路由，路由就是叠加通道，即在NCHW中的C进行叠加
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        #路由
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52

'''
测试
'''
if __name__ == '__main__':
    trunk = MainNet()

    #NCHW
    x = torch.Tensor(2, 3, 416, 416)

    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
    # print(y_13.view(-1, 3, 5, 13, 13).shape)


