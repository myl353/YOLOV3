# -*- coding: utf-8 -*-
'''
聚类得到各个尺寸的3种建议框
'''

def read_file(file_path):
    file=open(file_path)

    for line in file:
        # print(line)
        line=line.strip().split()
        # print(line)
        line1=line[1:]
        # print(line1)
        #[int(x) for x in line1]将内部的数据转为整数型
        #[i:i+5]将其按步长为5分割为单独的列表
        line2=[[int(x) for x in line1][i:i+5] for i in range(0,len(line1),5)]
        # print(line2)

    return

if __name__ == '__main__':
    file_path=r'data1/info.txt'
    read_file(file_path)