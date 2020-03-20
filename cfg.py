# -*- coding: utf-8 -*-

'''
存放一些信息；
'''

#图片的宽高
IMG_HEIGHT = 416
IMG_WIDTH = 416

#总的类别数
CLASS_NUM = 2

#三种尺度的3种建议框
ANCHORS_GROUP = {
    13: [[353, 306], [346, 228], [216, 275]],   #13尺度的建议框检测大目标
    26: [[150, 314], [123, 372], [150, 244]],   #26尺度的建议框检测中目标
    52: [[225, 158], [142, 112], [44, 92]]      #52尺度的建议框检测小目标
}

#各个建议框的面积，用来求iou的
ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
