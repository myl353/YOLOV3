# -*- coding: utf-8 -*-
import os
import json

names=os.listdir('data1/outputs')
# print(names)
img_information=open('data1/info.txt','w')

for i in range(len(names)):

    name=names[i]
    # print(name)

    file=open(os.path.join('data1/outputs',name))
    # print(file)
    data=json.load(file)
    coor = [f'{data["path"]}']
    # print(data['outputs']['object'])
    data=data['outputs']['object']
    for j in data:
        print(j)
        # print(j['name'])
        xmin = j['bndbox']['xmin']
        ymin = j['bndbox']['ymin']
        xmax = j['bndbox']['xmax']
        ymax = j['bndbox']['ymax']
        # print(xmin,ymin,xmax,ymax)
        cx=(xmax+xmin)//2
        cy=(ymax+ymin)//2
        # print(cx,cy)
        w=xmax-xmin
        h=ymax-ymin
        coor.extend([int(j['name']),cx,cy,w,h])
    # print(coor)
    #将列表中的数据一个一个写进去，不然又列表符号[]
    for line in coor:
        img_information.write(f'{line} ')
    img_information.write('\n')


    # break

