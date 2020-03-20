# -*- coding: utf-8 -*-
'''
测试器，调用训练好的yolov3网络;用的GPU测试

注意：测试的图片大小要和训练的图片大小一样，不然网络输出的特征图是不同的

输入任意大小的图片，将其填充+resize转为416*416的图片，传入网络，得到目标框，再在原图上画出检测框

'''


from YOLOV3.Two_target.module01 import *
from YOLOV3.Two_target import cfg
import torch
from PIL import Image,ImageDraw,ImageFont
from torchvision.transforms import transforms
from YOLOV3.Two_target.utils02 import nms

class Detector(torch.nn.Module):

    def __init__(self,model_path):
        super(Detector, self).__init__()

        self.model_path=model_path
        self.net = Darknet53().cuda()
        self.net.load_state_dict(torch.load(model_path))

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

        #三个尺度的所有框
        #detach()可以去掉，不影响，只是输出好看点
        boxes=torch.cat([boxes_13, boxes_26, boxes_52], dim=0).detach()


        final_boxes=nms(boxes,0.1)

        return final_boxes


    def _filter(self, output, thresh):
        #output原来是：NCHW，转为NHWC
        output = output.permute(0, 2, 3, 1)
        #将NHWC转为NHW,3,15 ,3表示各个尺度上的3种建议框，15：置信度+中心点+宽高+10分类
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        '''
        trainer04在训练损失时将置信度切了出来加了sigmiod激活的：
        sigmiod的作用是将其控制在0-1之间
        由于是在求损失的时候加的sigmiod，而不是网络层中，所以这里也需要加sigmoid(其实不加也可以，只有很小的影响)
        '''
        #...表示忽略前面所有维度，直达最后一维，0表示取最后一维的第一个值，即置信度(也可以说是iou)
        #output[..., 0] > thresh表示取output最后一维(就是置信度)大于阈值的索引
        #mask是一维的布尔值列表，形状是NHW3
        mask = torch.sigmoid(output[..., 0]) > thresh
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
        anchors = torch.Tensor(anchors).cuda()

        #idxs的形状是(M,4)，后面的4对应NHW3，所以取0就对应N,即对应第几张图片
        # 这里测试时是多张图片一起测试，所以才判断属于第几张图片，单张测试代码一样，因为数据形状都有N
        n = idxs[:, 0]  # 所属的图片

        #3对应NHW3中的3，即是第几种建议框
        a = idxs[:, 3]  # 建议框

        '''
        trainer04在训练损失时将中心点切了出来加了sigmiod激活的：
        sigmiod的作用是将其控制在0-1之间
        由于是在求损失的时候加的sigmiod，而不是网络层中，所以这里也需要加sigmoid
        '''
        # t是尺度相对原图的缩放比例
        #idxs[:, 1]对应NHW3中的H,vecs[:, 2]对应NHW3,15中的W
        #由于做标签时，已经把H和W进行了换位，所以这里vecs[:, 2]对应的形状应该是NWH3,15,即对应的是H
        #idxs[:, 1]是原图中的整数部分，vecs[:, 2]是原图中的小数部分
        cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 2])) * t  # 原图的中心点y
        #idxs[:, 2]取的是NHW3中的W,vecs[:, 1]取的是置信度+中心点的高+中心点的宽+宽高+10分类中的W(中心点的宽)
        cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 1])) * t  # 原图的中心点x
        '''注：idxs取的是NHW3中对应的维度的索引，而vecs取的是15中对应的数值，所以idxs的宽高还是HW,而vecs在做标签时换了，所以是WH'''

        #反算宽和高
        #anchors[a]得到对应的建议框，0是建议框的宽，vecs[:, 3]是宽的偏移量
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])
        # print('w.shape:',w.shape)

        #置信度
        #其实这里的置信度都可以不加sigmiod了，因为这里是用来NMS排序的，不加sigmiod也不会影响顺序
        cond=torch.sigmoid(vecs[:,0])

        #所属类别
        try:
            cls=torch.argmax(torch.softmax(vecs[:,5:],dim=1),dim=1).float()
            # print('cls:', cls)
            # print('cls.shape:', cls.shape)
        except:
            cls=torch.Tensor([]).cuda()
            # print(cls.size())
            print('这个尺度检测到的中心点置信度都不满足要求')


        #[n.float(), cond,cx, cy, w, h,cls]是列表，所以要用torch.stack()
        #dim=1表示按1维进行拼接
        #所属图片、置信度、中心点、宽高、类别
        return torch.stack([n.float(), cond,cx, cy, w, h,cls], dim=1)

    #将图片按长填充为正方形，并resize为416*416
    def resize_img(self,image):
        w,h=image.size
        # 最大边
        max_size = max(w, h)

        #缩放比例
        _scale=max_size/416

        # 以最长边为边长创建新的正方形图片，背景为白色(255,255,255)，黑色是(0,0,0)
        new_img = Image.new('RGB', (max_size, max_size), color=(255, 255, 255))
        # 将img由在中间粘贴
        new_img.paste(image, box=(int((max_size - w) / 2), int((max_size - h) / 2)))
        # new_img.show()
        new_img = new_img.resize((416, 416),Image.ANTIALIAS)
        return new_img,_scale


if __name__ == '__main__':
    font = ImageFont.truetype('STKAITI.TTF', size=20)
    cls_dict = {0: '人', 1: '狗'}

    detector = Detector(model_path=r'models/yolov3_two04.pth')
    img_path='data/1 (13).jpg'
    image=Image.open(img_path)
    w_,h_=image.size
    img,scale=detector.resize_img(image=image)
    img_tensor=transforms.ToTensor()(img).unsqueeze(0).cuda()
    # print(img_tensor)
    y = detector(img_tensor, 0.6, cfg.ANCHORS_GROUP)
    print(y)
    color=['red','green']

    #在resize后的图上画框
    draw=ImageDraw.Draw(img)
    #在原图上画框
    draw1=ImageDraw.Draw(image)
    for box in y:
        x1=int(box[2]-box[4]/2)
        y1=int(box[3]-box[5]/2)
        x2=int(box[2]+box[4]/2)
        y2=int(box[3]+box[5]/2)

        draw.rectangle((x1,y1,x2,y2),outline=color[int(box[6])],width=2)
        draw.text((x1, y1), text=str(cls_dict[int(box[6])]) + ' : ' + str(round(float(box[1]), 4)), font=font,fill=(0, 255, 255))

        # if w_ > h_:
        #     #求在原图上的坐标
        #     x1_ = x1 * scale
        #     y1_ = y1 * scale - (w_-h_)//2
        #     x2_ = x2 * scale
        #     y2_ = y2 * scale - (w_-h_)//2
        #
        #     draw1.rectangle((x1_, y1_, x2_, y2_), outline=color[int(box[6])], width=2)
        # elif w_<=h_:
        #     x1_ = x1 * scale - (h_-w_) // 2
        #     y1_ = y1 * scale
        #     x2_ = x2 * scale - (h_-w_) // 2
        #     y2_ = y2 * scale
        #     draw1.rectangle((x1_, y1_, x2_, y2_), outline=color[int(box[6])], width=2)

        #求在原图上的坐标，将上面的判断语句合并为1句
        max_size=max(w_,h_)

        #resize中填充时就相当于在短边加了一个(max_size-w_)//2或(max_size-h_)//2，所以这里要减去这个，才是原坐标
        x1_ = x1 * scale - (max_size-w_)//2
        y1_ = y1 * scale - (max_size-h_)//2
        x2_ = x2 * scale - (max_size-w_)//2
        y2_ = y2 * scale - (max_size-h_)//2
        draw1.rectangle((x1_, y1_, x2_, y2_), outline=color[int(box[6])], width=2)
        draw1.text((x1_, y1_), text=str(cls_dict[int(box[6])]) + ' : ' + str(round(float(box[1]), 4)), font=font,fill=(0, 255, 255))

    img.show()
    image.show()
    img.save('xx.jpg')








