# -*- coding: utf-8 -*-
'''
训练器，训练yolov3网络,用GPU训练网络
'''

from YOLOV3.Two_target import dataset01
from YOLOV3.Two_target.module01 import *
import os
import torch


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

    # print(output[mask_obj].shape,target[mask_obj].shape)
    #正样本损失
    loss_obj = torch.mean((output[mask_obj].float() - target[mask_obj].float()) ** 2)
    #负样本损失，这里的负样本损失相当于4个都求了
    # print(output[mask_noobj], target[mask_noobj])
    loss_noobj = torch.mean((output[mask_noobj].float() - target[mask_noobj].float()) ** 2)
    #总损失，之所以将两个损失分开，是因为数据中正样本太少，负样本太多，所以分开是为了加个alpha
    #alpha一般给的较大,如0.8、0.9，使正样本的损失占比重更大；
    #具体的根据目标数来设置，目标越多，alpha越小，目标越少，alpha就要给越大；使网络训练正负方样本时保持均衡
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    '''
    注：这里是 置信度、中心点、宽高、分类都一起进行训练，就相当于4个都是用均方差来求的损失
    可以将置信度和分类切出来，用交叉熵来求损失，训练速度会更快，
    但精度会不会变好不清楚，但肯定不会更差
    '''
    return loss


if __name__ == '__main__':

    model_path='models/yolov3_two.pth'

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #获取数据
    myDataset = dataset01.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    #实例化网络
    net = Darknet53().to(device)
    # net.train()

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    opt = torch.optim.Adam(net.parameters())

    i=0
    while True:
        i+=1
        for target_13, target_26, target_52, img_data in train_loader:
            target_13, target_26, target_52, img_data=target_13.to(device), target_26.to(device), target_52.to(device), img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"epochs--{i}--{loss.item()}")
        torch.save(net.state_dict(),model_path)
