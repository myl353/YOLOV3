# -*- coding: utf-8 -*-
'''
训练器，训练yolov3网络
用GPU训练网络
将置信度、中心点、宽高、分类全切出来分别求损失
置信度用BCELoss；加sigmoid;用正样本和负样本
中心点用MSELoss;加sigmoid；只用正样本
宽高用MSELoss；只用正样本
分类用NLLLoss;加softmax；只用正样本

'''

from YOLOV3.Two_target import dataset02
from YOLOV3.Two_target.module01 import *
import os
import torch
import torch.nn as nn


def loss_fn(output, target, alpha):
    '''将输出与标签样式对应，NHW3,,15'''
    #output输出形状是:NCHW，转为NHWC
    output = output.permute(0, 2, 3, 1)
    #再将这里转为：N,H,W,3,7  3表示每个尺度的3种建议框，7表示 置信度(1)+中心点(2)+宽高(2)+2分类(2)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    #...表示忽略前面所有维度，直接到最后一维，就是那个15；target[..., 0] 就取出了所有的置信度
    #target[..., 0] > 0获取所有置信度大于0的索引
    mask_obj = target[..., 0] > 0
    #target[..., 0] == 0获取所有置信度等于0的索引
    mask_noobj = target[..., 0] == 0

    '''
    置信度损失，用BCELoss；输出用sigmoid将其压缩到0-1之间;
    由于是在损失加的sigmiod,而不是网络层中，所以在测试时，也切出来需要加sigmiod
    '''
    out_iou_obj=torch.sigmoid(output[mask_obj][:,0].float())
    target_iou_obj=target[mask_obj][:,0].float()
    iou_loss_obj=bceloss(out_iou_obj,target_iou_obj)
    # print(iou_loss_obj)

    out_iou_noobj=torch.sigmoid(output[mask_noobj][:,0].float())
    target_iou_noobj=target[mask_noobj][:,0].float()
    iou_loss_noobj=bceloss(out_iou_noobj,target_iou_noobj)
    # print(iou_loss_noobj)


    '''
    中心点损失，用MSELoss；只用正样本；输出用sigmiod将其压缩到0-1之间(原文就是这样的)；
    由于是在损失加的sigmiod,而不是网络层中，所以在测试时，也切出来需要加sigmiod
    '''
    out_center_obj=torch.sigmoid(output[mask_obj][:,1:3].float())
    target_center_obj=target[mask_obj][:,1:3].float()
    center_loss_obj=mseloss(out_center_obj,target_center_obj)
    # print(center_loss_obj)

    #负样本的中心点损失，不使用
    # out_center_noobj = torch.sigmoid(output[mask_noobj][:, 1:3].float())
    # target_center_noobj = torch.sigmoid(target[mask_noobj][:, 1:3].float())
    # center_loss_noobj = loss_fuc2(out_center_noobj, target_center_noobj)
    # print(center_loss_noobj)

    '''宽高损失，用MSELoss；只用正样本'''
    out_wh_obj = output[mask_obj][:, 3:5].float()
    target_wh_obj = target[mask_obj][:, 3:5].float()
    wh_loss_obj = mseloss(out_wh_obj, target_wh_obj)
    # print(wh_loss_obj)


    '''
    分类损失，用NLLLLoss或softmax交叉熵loss；只用正样本
    注：用NLLLLoss则对输出加softmax，用softmax交叉熵则不用，它自带了softmax
    '''
    out_cls_obj=torch.softmax(output[mask_obj][:,5:],dim=1)
    # 由于制作样本时对应的分类标签使用的是one-hot，而传入交叉熵损失和Nullloss的是直接的标签形式，所以要用torch.argmax(target[mask_obj][:,5:],dim=1)来求出具体类别才行
    target_cls_obj=target[mask_obj][:,5].long()
    cls_loss_obj=nllloss(out_cls_obj,target_cls_obj)
    # print(cls_loss_obj)

    '''
    总的损失,
    由于iou_loss_obj、center_loss_obj、wh_loss_obj、cls_loss_obj都是正样本得到的损失，所以将他们全加在一起
    '''
    loss=alpha*(iou_loss_obj + center_loss_obj + wh_loss_obj + cls_loss_obj) + (1-alpha)*iou_loss_noobj

    return loss


if __name__ == '__main__':

    model_path='models/yolov3_two04_01_02.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #获取数据
    myDataset = dataset02.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    #实例化网络
    net = Darknet53().to(device)
    # net.train()

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    opt = torch.optim.Adam(net.parameters())

    bceloss=nn.BCELoss()
    mseloss=nn.MSELoss()
    nllloss=nn.NLLLoss()

    i=0
    while True:
        i+=1
        for target_13, target_26, target_52, img_data in train_loader:
            target_13, target_26, target_52, img_data = target_13.to(device), target_26.to(device), target_52.to(device), img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13, target_13, 0.6)
            loss_26 = loss_fn(output_26, target_26, 0.6)
            loss_52 = loss_fn(output_52, target_52, 0.6)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"epochs--{i}--{loss.item()}")
        torch.save(net.state_dict(),model_path)
