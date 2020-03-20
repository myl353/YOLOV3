# -*- coding: utf-8 -*-
'''
测试一些不常用方法的效果
'''

import torch
import math
import numpy as np

# a = torch.Tensor([1, 2, 3, 4])
# b = a < 3  # mask
# print(b)
# print(a[b])
# print(b.nonzero())
# print(a[b.nonzero()])

# a = torch.Tensor([[1, 2], [5, 6], [3, 1], [2, 8]])
# a=torch.arange(2*3*4*5).reshape(2,3,4,5)
# print(a.shape)
# # b = a < 3
# # print(b)
# # print(a[b])
# b = a[..., 0] > 5
# print(b.shape)
# print(a[b])
# print(a[b][:,0])
# print(b.nonzero())

# print(math.modf(3.4))

#NHW3,15    ,一共480,2*4*4*3=96
a=torch.arange(2*4*4*3*5).reshape(2,4,4,3,5)


def _filter(output, thresh):

    # ...表示忽略前面所有维度，直达最后一维，0表示取最后一维的第一个值，即置信度(也可以说是iou)
    # output[..., 0] > thresh表示取output最后一维(就是置信度)大于阈值的索引
    # mask是一维的布尔值列表，形状是NHW3
    mask = output[..., 0] > thresh
    print(mask.shape)
    # 取出mask中不为0的索引，也就是output中置信度大于thresh的数据的具体索引，形状与mask不一样
    # 形状是(47,4),前面的47表示output中置信度大于阈值的数据个数，4表示由0-3维满足要求的数据的索引，即NHW3对应的索引
    idxs = mask.nonzero()
    print(idxs.shape)

    # 取出output中置信度大于thresh的数据，这里是按布尔列表取值
    #形状为(47,5)，前面的47表示output中置信度大于阈值的数据个数，5就是NHW3,5中最后一维的5
    vecs = output[mask]
    print(vecs)

    # return idxs, vecs

print(_filter(a,240))


def _parse(self, idxs, vecs, t, anchors):
    # 各个尺度的3种建议框
    anchors = torch.Tensor(anchors)

    # idxs的形状是(M,4)，后面的4对应NHW3，所以取0就对应N,即对应第几张图片
    # 这里测试时是多张图片一起测试，所以才判断属于第几张图片，单张测试代码一样，因为数据形状都有N
    n = idxs[:, 0]  # 所属的图片

    # 3对应NHW3中的3，即是第几种建议框
    a = idxs[:, 3]  # 建议框

    # t是尺度相对原图的缩放比例
    # idxs[:, 1]对应NHW3中的H,vecs[:, 2]对应NHW3,15中的W
    # 由于做标签时，已经把H和W进行了换位，所以这里vecs[:, 2]对应的形状应该是NWH3,15,即对应的是H
    # idxs[:, 1]是原图中的整数部分，vecs[:, 2]是原图中的小数部分
    cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
    # idxs[:, 2]取的是NHW3中的W,vecs[:, 1]取的是置信度+中心点的高+中心点的宽+宽高+10分类中的W(中心点的宽)
    cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x
    '''注：idxs取的是NHW3中对应的维度的索引，而vecs取的是15中对应的数值，所以idxs的宽高还是HW,而vecs在做标签时换了，所以是WH'''

    # 反算宽和高
    # anchors[a]得到对应的建议框，0是建议框的宽，vecs[:, 3]是宽的偏移量
    w = anchors[a, 0] * torch.exp(vecs[:, 3])
    h = anchors[a, 1] * torch.exp(vecs[:, 4])

    # [n.float(), cx, cy, w, h]是列表，所以要用torch.stack()
    # dim=1表示按1维进行拼接
    return torch.stack([n.float(), cx, cy, w, h], dim=1)


