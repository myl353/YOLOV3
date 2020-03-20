# -*- coding: utf-8 -*-
'''
测试一些不常用方法的效果
'''

import torch
import math
import numpy as np



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
    print(vecs.shape)

    return idxs, vecs

# print(_filter(a,240))
