# -*- coding: utf-8 -*-
'''
测试器，调用训练好的yolov3网络；用的cpu测试

注意：测试的图片大小要和训练的图片大小一样，不然网络输出的特征图是不同的

这个代码少了几个部分：
1、_parse反算原图后没有输出类别，所以只能做单目标，做多目标要加上分类
2、这个代码没有导入NMS来筛选框，每个目标上都有多个尺度的多种建议框，需要进行筛选
'''


from YOLOV3.Two_target.module01 import *
from YOLOV3.Two_target import cfg
import torch
from PIL import Image,ImageDraw
from torchvision.transforms import transforms


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53().cpu()
        self.net.load_state_dict(torch.load('models/yolov3_two04.pth'))

        self.net.eval()

    def forward(self, input, thresh, anchors):
        #网络对3个尺度的输出
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        # print('vecs_13:',vecs_13)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])
        # print('boxes_13:',boxes_13)

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        # print('vecs_26:',vecs_26)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        # print('vecs_52:',vecs_52)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)


    def _filter(self, output, thresh):
        #output原来是：NCHW，转为NHWC
        output = output.permute(0, 2, 3, 1)
        #将NHWC转为NHW,3,15 ,3表示各个尺度上的3种建议框，15：置信度+中心点+宽高+10分类
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        #...表示忽略前面所有维度，直达最后一维，0表示取最后一维的第一个值，即置信度(也可以说是iou)
        #output[..., 0] > thresh表示取output最后一维(就是置信度)大于阈值的索引
        #mask是一维的布尔值列表，形状是NHW3
        mask = output[..., 0] > thresh
        #取出mask中不为0的索引，也就是output中置信度大于thresh的数据的具体索引，形状与mask不一样
        #形状是(M,4),前面的M表示output中置信度大于阈值的数据个数，4表示由0-3维满足要求的数据的索引，即NHW3对应的索引
        idxs = mask.nonzero()

        #取出output中置信度大于thresh的数据，这里是按布尔列表取值
        #形状为(M,15)，前面的M表示output中置信度大于阈值的数据个数，15就是NHW3,15中最后一维的15个数据
        # 即 置信度+中心点+宽高+10分类
        #但由于制作标签时将H和W进行了换位，所以vecs中的15对应的形状应该是置信度+中心点的高+中心点的宽+宽高+10分类
        vecs = output[mask]
        # print(vecs)

        return idxs, vecs

    #反算原图
    def _parse(self, idxs, vecs, t, anchors):
        #各个尺度的3种建议框
        anchors = torch.Tensor(anchors)

        #idxs的形状是(M,4)，后面的4对应NHW3，所以取0就对应N,即对应第几张图片
        # 这里测试时是多张图片一起测试，所以才判断属于第几张图片，单张测试代码一样，因为数据形状都有N
        n = idxs[:, 0]  # 所属的图片

        #3对应NHW3中的3，即是第几种建议框
        a = idxs[:, 3]  # 建议框

        '''
        这里没加sigmiod，原文中是有sigmiod的；
        可以在网络层中对中心点的宽和高输出加sigmiod(但这个面向对象式的网络写法没法加)，在网络层中加了这里就不需要加了；
        也可以在训练时，将中心点的宽和高点单独切出来，使用sigmiod然后做损失，但这种方式这里最好也加上；
        其实这里不加也可以，训练好了网络后，不会影响结果，但加了训练速度会更快
        '''
        # t是尺度相对原图的缩放比例
        #idxs[:, 1]对应NHW3中的H,vecs[:, 2]对应NHW3,15中的W
        #由于做标签时，已经把H和W进行了换位，所以这里vecs[:, 2]对应的形状应该是NWH3,15,即对应的是H
        #idxs[:, 1]是原图中的整数部分，vecs[:, 2]是原图中的小数部分
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        #idxs[:, 2]取的是NHW3中的W,vecs[:, 1]取的是置信度+中心点的高+中心点的宽+宽高+10分类中的W(中心点的宽)
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x
        '''注：idxs取的是NHW3中对应的维度的索引，而vecs取的是15中对应的数值，所以idxs的宽高还是HW,而vecs在做标签时换了，所以是WH'''

        #反算宽和高
        #anchors[a]得到对应的建议框，0是建议框的宽，vecs[:, 3]是宽的偏移量
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        #[n.float(), cx, cy, w, h]是列表，所以要用torch.stack()
        #dim=1表示按1维进行拼接
        return torch.stack([n.float(), cx, cy, w, h], dim=1)


if __name__ == '__main__':
    detector = Detector()
    img_path='data1/images/0.jpg'
    img=Image.open(img_path)
    img_tensor=transforms.ToTensor()(img).unsqueeze(0)
    # print(img_tensor)
    y = detector(img_tensor, 0.3, cfg.ANCHORS_GROUP)
    # print(y)

    # draw=ImageDraw.Draw(img)






