# -*- coding: utf-8 -*-
'''
yolov3数据集制作
在制作分类标签时直接用具体类别，而不用one-hot形式，这样求分类损失(使用交叉熵、NLLLoss)时更简单
'''

import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from YOLOV3.Two_target import cfg
import os

from PIL import Image
import math

#存放每张图片名及标签的文件
# 图片文件夹+图片名+目标类别+中心点+宽高+目标类别+中心点+宽高+....
#形如：images/1.jpg 1 12 13 51 18 2 22 31 55 98 2 44 33 62 62
LABEL_FILE_PATH = "data1/info.txt"
#存放图片文件夹和标签文件的文件夹
# IMG_BASE_DIR = "data1"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()  #转张量、归一化、图片形状转为HWC
])

#cls_num是类别数，v是当前目标的类别，做类别的one-hot
#分类标签直接用具体类别，这里就不需要这个了
# def one_hot(cls_num, v):
#     b = np.zeros(cls_num)
#     b[v] = 1.
#     # print(b)
#     return b


class MyDataset(Dataset):

    def __init__(self):
        #打开标签文件
        with open(LABEL_FILE_PATH) as f:
            #读取标签文件
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        #存放所以的标签
        labels = {}

        line = self.dataset[index]
        #分割每一列
        strs = line.split()
        #打开图片，PIL打开为WHC
        #strs[0]保存的是图片的绝对路径，不需要使用os.path.join()了
        _img_data = Image.open(strs[0])
        # #将图片转为416*416的
        # #一般制作数据集是先转换尺寸再标记中心点和宽高，所以这里就不需要了
        # _img_data=_img_data.resize((416,416))
        #转为HWC，且做了归一化和转张量
        img_data = transforms(_img_data)

        #读取第一列之后的数据，即目标类别、中心点和宽高，每张图片可能有多个目标
        _boxes = np.array([float(x) for x in strs[1:]])
        # _boxes = np.array(list(map(float, strs[1:])))

        #np.split将np数据切分,得到所有目标的类别、中心点和宽高，len(_boxes) // 5表示分为几块(即一个目标一块)
        boxes = np.split(_boxes, len(_boxes) // 5)

        #feature_size是13/26/52  anchors是每个尺度对应的三个建议框。
        #对各个尺度都要做标签
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            #创建标签存放的形状
            #标签的形状：{feature_size：[]}，中括号中数据的形状是：feature_size,feature_size,3,5+cfg.CLASS_NUM
            #中括号中数据的形状例：13,13,3,6    13*13表示13*13的输出尺度，这里与输出是对应的，3表示每个尺度的3种建议框，6表示 置信度+中心点+宽高+类别
            #网络输出的形状是N,13,13,18  ,所以做损失是要注意形状的变换
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + 1))

            #循环读取一张图片的每个目标的标签数据
            for box in boxes:
                #目标的类别、目标的中心点坐标的x和y、目标的宽和高
                cls, cx, cy, w, h = box
                #math.modf将浮点数分为小数部分和整数部分，cx_offset是小数部分，cx_index是整数部分
                #cx * feature_size / cfg.IMG_WIDTH 其实就是cx/32，cx/16，cx/8 ，对应三种尺度建议框大的缩放比例
                # cx_index就是将原图分成了cx/32份(13)，cx/16(26)，cx/8(52)份后，中心点所在的位置的索引，cx_offset就是中心点在那一部分内部的偏移量
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)

                #遍历每个尺度的每个建议框，对应每个建议框做标签
                for i, anchor in enumerate(anchors):
                    #建议框的面积
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]

                    #宽高的偏移量计算
                    #实际框的宽(w)/建议框的宽(anchor[0]),实际框的高(h)/建议框的高(anchor[1])
                    #log在最终加上标签时加了
                    p_w, p_h = w / anchor[0], h / anchor[1]

                    #实际框的面积，用来求iou
                    p_area = w * h
                    #求iou,是用建议框和实际框中的较小面积比上建议框和实际框中的较大面积
                    #之所以用这种方式，因为yolov3中对同一目标的框都默认是同心的，即每个框的中心点都一样
                    #这里iou就作为置信度
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    #最终的标签数据
                    #int(cy_index), int(cx_index)就是对应在13*13(26*26、52*52)每个区域的索引
                    #i表示每种尺度的是第几个建议框，每种尺度都有3个建议框
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), int(cls)])

        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    dataset=MyDataset()
    dataset[0]
