# -*- coding: utf-8 -*-
'''
yolov3要使用的一些工具，如iou、nms

nsm改为了softnms
'''

'''
如果是传统的NMS操作，那么当B中的box bi和M的IOU值大于阈值Nt，那么就从B和S中去除该box；
如果是Soft NMS，则对于B中的box bi也是先计算其和M的IOU，然后该IOU值作为函数f()的输入，最后和box bi的score si相乘作为最后该box bi的score。

'''
import torch
from YOLOV3.Two_target import cfg


def iou(box, boxes, mode="inter"):
    #完整数据是：所属图片(1)、置信度(1)、中心点(2)、宽高(2)、类别(1)
    box_cx,box_cy,box_w,box_h=box[2],box[3],box[4],box[5]
    box_x1,box_y1,box_x2,box_y2=(box_cx-box_w/2),(box_cy-box_h/2),(box_cx + box_w/2),(box_cy + box_h/2)
    # print(box_x1,box_y1,box_x2,box_y2)
    box_area = box_w * box_h
    # print(box_area)


    boxes_area = boxes[:,4] *boxes[:,5]
    print('boxes_area.shape:',boxes_area.shape)

    # print(3//2)
    # print((boxes[:, 2]-boxes[:,3]//2))

    x1 = torch.max(box_x1, (boxes[:, 2]-boxes[:,4]/2))
    y1 = torch.max(box_y1, (boxes[:, 3]-boxes[:,5]/2))
    x2 = torch.min(box_x2, (boxes[:, 2]+boxes[:,4]/2))
    y2 = torch.min(box_y2, (boxes[:, 3]+boxes[:,5]/2))

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter)
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)


#这个NMS仅循环了类别，没有循环属于哪张图，适合一张一张检测的时候用
def nms(boxes, thresh,cls_num=cfg.CLASS_NUM, mode='inter',method=1,sigma=0.5,threshold=0.4):
    keep_boxes=[]
    # print(type(cls_num))
    for i in range(cls_num):
        mask= boxes[:,-1]== i
        # print(boxes[mask])

        args =boxes[mask][:,1].argsort(descending=True)
        sort_boxes=boxes[mask][args]
        # print('sort_boxes:',sort_boxes)

        while len(sort_boxes) > 0:
            _box = sort_boxes[0]
            keep_boxes.append(_box)

            if len(sort_boxes) > 1:
                _boxes = sort_boxes[1:]
                # print('_boxes.shape:',_boxes.shape)
                _iou = iou(_box, _boxes, mode)
                # print('_iou.shape:',_iou.shape)
                # #只有iou小于阈值的才会留下，其它的不要了
                # sort_boxes = _boxes[_iou < thresh]

                # weight=torch.ones(_iou.shape)

                #线性加权，仅对_iou>thresh的进行加权
                if method == 1:
                        weight = 1 - _iou[_iou>thresh]
                #高斯加权，是对所有的框都进行了加权
                elif method ==2:
                        weight = torch.exp(-(_iou**2)/sigma)
                #不加权，直接丢
                else:
                    weight=_iou[_iou > thresh]*0

                # print('_boxes[:,1]:',_boxes[:,1])
                _boxes[:,1] = weight*_boxes[:,1]
                # print('_boxes[:,1]:',_boxes[:,1])

                mask02 = _boxes[:, 1] > threshold
                # print('mask02:',mask02)
                # print(mask02)
                sort_boxes=_boxes[mask02]
                # print('sort_boxes:',sort_boxes)
            else:
                break

    return keep_boxes


# def detect(feature_map, thresh):
#     masks = feature_map[:, 4, :, :] > thresh
#     idxs = torch.nonzero(masks)


if __name__ == '__main__':
    ##所属图片、置信度、中心点、宽高、类别：[n.float(), cond, cx, cy, w, h, cls]
    # cls_num=2

    box = torch.Tensor([0,2, 2, 3, 3, 6,0])

    boxes = torch.Tensor([[0,1, 2, 3, 3, 6,1], [0,2, 2, 4, 4, 5,0], [0,3, 2, 5, 5, 4,0]])
    # print(iou(box, boxes, mode="inter"))
    print(nms(boxes, 0.5))
    # import numpy as np
    #
    # a = np.array([[1, 2], [3, 4]])
    # print(a[:, 1])

