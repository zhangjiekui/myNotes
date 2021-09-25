# -*- coding: utf-8 -*-
# @Time : 2021/9/24 9:30
# @Author : zjk
# @File : f1_data_file_process.py
# @Project : YOLOX
# @IDE :PyCharm

import xml.etree.ElementTree as ET
import os
import cv2 as cv
import numpy as np
import torch
from yolov1.my_yolov1_model import Yolov1
from yolov1.my_yolov1_loss import YoloLoss

base_dir = r"C:\winyolox\VOC2007"
sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}

classes = list(classes_dict.keys())

def convert_annotation(year, image_id, f, base_dir=base_dir):
    in_file = os.path.join('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    in_file = os.path.join(base_dir, in_file)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # classes = list(classes_num.keys())
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))
        # ' 156,97,351,270,6'  bbox坐标和类别id

def get_img_and_its_bbox_cls(list_of_datasets=sets):
    for year, image_set in list_of_datasets:
        datasets_split = os.path.join('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set))
        datasets_split = os.path.join(base_dir, datasets_split)
        with open(datasets_split, 'r') as f:
            image_ids = f.read().strip().split()
        datasets_split_result = os.path.join("VOCdevkit", '%s_%s.txt' % (year, image_set))
        datasets_split_result = os.path.join(base_dir, datasets_split_result)
        print(f"dataset:[{year},{image_set}],读取源文件:{datasets_split}, 写入目标文件:{datasets_split_result}")

        with open(datasets_split_result, 'w') as f:
            for image_id in image_ids:
                img_path = '%s/VOC%s/JPEGImages/%s.jpg' % ("VOCdevkit", year, image_id)
                f.write(img_path)
                convert_annotation(year, image_id, f, base_dir=base_dir)  # 逐一遍历所有xml标注文件，提取所需信息，并接上行写入文件
                f.write('\n')
    print("完成voc2007原始数据集的转换！得到【图片文件 bbox（s）坐标和类别】形如：‘VOCdevkit/VOC2007/JPEGImages/000017.jpg 185,62,279,199,14 90,78,403,336,12’")


def get_train_val_XY(base_dir=base_dir):
    train_datasets = []
    val_datasets = []
    train_txt_path = os.path.join(base_dir, "VOCdevkit", '2007_train.txt')
    val_txt_path = os.path.join(base_dir, "VOCdevkit", '2007_val.txt')
    with open(train_txt_path, 'r') as f:
        train_datasets = train_datasets + f.readlines()
    with open(val_txt_path, 'r') as f:
        val_datasets = val_datasets + f.readlines()
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    for item in train_datasets:
        item = item.replace("\n", "").split(" ")
        X_train.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_train.append(arr)
    for item in val_datasets:
        item = item.replace("\n", "").split(" ")
        X_val.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_val.append(arr)

    print(f"X为图片文件列表['VOCdevkit/VOC2007/JPEGImages/000012.jpg',...]，Y为[['116,167,360,400,18', '141,153,333,229,18'],...]")
    print(f"   Train->X:{len(X_train)},Y:{len(Y_train)}")
    print(f"   Valid->X:{len(X_val)},Y:{len(Y_val)}")
    return X_train,Y_train, X_val, Y_val

def read(image_path, label, i, S=7,base_dir=base_dir):
    if base_dir is not None:
        image_path = os.path.join(base_dir, image_path)
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_h, image_w = image.shape[0:2]
    image = cv.resize(image, (448, 448))
    image = image / 255.

    label_matrix = np.zeros([7, 7, 30])
    for l in label[:2]:
        # 最多只取两个label:即label[:2]
        l = l.split(',')
        l = np.array(l, dtype=int)
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        cls = l[4]
        x = (xmin + xmax) / 2 / image_w  # 求原始中心点x坐标，并除以宽度，归一化
        y = (ymin + ymax) / 2 / image_h  # 求原始中心点y，并除以高度，归一化
        w = (xmax - xmin) / image_w  # 宽度归一化
        h = (ymax - ymin) / image_h  # 高度归一化
        loc = [S * x, S * y]  # S*S个Cell
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i  # 相对于Cell左上角的偏移，归一化到[0,1]
        x = loc[0] - loc_j
        assert x>=0
        assert y >= 0
        if x<0 or y<0:
            print("出现了负数")
            raise Exception

        if label_matrix[loc_i, loc_j, 24] == 1 and label_matrix[loc_i, loc_j, 29] == 0:
            tips = "没有值"
            if label_matrix[loc_i, loc_j, cls] == 1.:
                tips = "已有值"

            label_matrix[loc_i, loc_j, cls] = 2
            print(f"   label_matrix[{loc_i}, {loc_j}, {cls}] = {label_matrix[loc_i, loc_j, cls]}")
            label_matrix[loc_i, loc_j, 25:29] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 29] = 1  # response
            print(f"序号({i}) label_matrix[{loc_i}, {loc_j}] "+image_path + tips + str(list(label_matrix[loc_i, loc_j])))

        if label_matrix[loc_i, loc_j, 24] == 0:
            label_matrix[loc_i, loc_j, cls] = 1
            label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 24] = 1  # response
    # print(f"  返回的image.shape{image.shape},label_matrix.shape{label_matrix.shape}，都已经归一化了。\n"
    #       f"    label_matrix[S,S,0:20]为物体类别的one-hot编码，如果[S,S,cls]=1,表明bbox1是对应这个类别，如果[S,S,cls]=2，表明bbox2是对应这个类别，\n"
    #       f"    如果[S,S]内有两个同类对象(即bbox1、bbox2都落在同一个Cell[S,S]，且是同一类别cls，则只有[S,S,cls]=2，没有[S,S,cls]=1)，方便分辩。\n"
    #       f"    label_matrix[S,S,20:24]为bbox1的坐标，label_matrix[S,S,24]=1标识bbox1落在[S,S]内；\n"
    #       f"    label_matrix[S,S,25:29]为bbox2的坐标，label_matrix[S,S,29]=1标识bbox2落在[S,S]内；\n")
    return image, label_matrix

def test_one_example(X_train,Y_train,i=10):
    print("##########测试网络及LOSS##########"
          "    同一cell[3, 2] 中两个不同物体,类别序号及顺序为[8,7] {'cat': 7, 'chair': 8}"
          "    其中label_matrix[3,2,7]=2，表明是bbox2里的对象为类别'cat': 7"
          "    其中label_matrix[3,2,8]=1，表明是bbox1里的对象为类别'chair': 8")
    # 同一cell[3, 2] 中两个不同物体,类别序号及顺序为[8,7] {'cat': 7, 'chair': 8}
    image, label_matrix = read(X_train[i], Y_train[i], i)
    image = torch.from_numpy(image)
    image = image.to(torch.float32)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, 0)
    label_matrix = torch.from_numpy(label_matrix).to(torch.float32)
    label_matrix = torch.unsqueeze(label_matrix, 0)
    net = Yolov1()
    loss_fun = YoloLoss()
    predictions = net(image)
    loss = loss_fun(predictions, label_matrix)
    return loss

def test_one_batch(X_train,Y_train,batch=[10,15]):
    print("##########测试网络及LOSS##########\n"
          
          "图片[10] 同一cell[3, 2] 中两个不同物体,\n"
          "图片[15] cell[3, 3]、[4,3]各有1个物体"
          )
    images = []
    label_matrixs = []
    for i in batch:
        image, label_matrix = read(X_train[i], Y_train[i], i)
        images.append(image)
        label_matrixs.append(label_matrix)
    image=torch.tensor(images)
    label_matrix=torch.tensor(label_matrixs)

    # image = torch.from_numpy(image)

    image = torch.permute(image, (0,3,1,2))
    # image = torch.unsqueeze(image, 0)
    # label_matrix = torch.from_numpy(label_matrix).to(torch.float32)
    # label_matrix = torch.unsqueeze(label_matrix, 0)
    image = image.to(torch.float32)
    label_matrix = label_matrix.to(torch.float32)
    assert (label_matrix<0).sum()==0
    net = Yolov1()
    loss_fun = YoloLoss()
    assert (label_matrix < 0).sum() == 0
    loss_pre = loss_fun(label_matrix, label_matrix)
    assert (label_matrix < 0).sum() == 0
    predictions = net(image)
    loss = loss_fun(predictions, label_matrix)
    assert (label_matrix < 0).sum() == 0
    loss_pre = loss_fun(label_matrix, label_matrix)

    return loss,loss_pre

if __name__ == '__main__':
    # get_img_and_its_bbox_cls()
    X_train,Y_train, X_val, Y_val=get_train_val_XY()
    # imgs_x = []
    # lbls_y = []
    # for i in [10, 15]:
    #     image, label_matrix=read(X_train[i], Y_train[i], i)
    #     imgs_x.append(image)
    #     lbls_y.append(label_matrix)

    # for i in range(50):
    #     image, label_matrix=read(X_train[i], Y_train[i], i)
    #     imgs_x.append(image)
    #     lbls_y.append(label_matrix)

    # loss = test_one_example(X_train=X_train,Y_train=Y_train,i=10)
    # print(loss)
    loss,loss_pre = test_one_batch(X_train, Y_train, batch=[10, 15])  #【10】[0,3,2]  # 【15】[1,3,3],exists_box[1,4,3]
    # exists_box[1,3,3]+exists_box[1,4,3]+exists_box[0,3,2]
    print(loss,loss_pre)
    print('end')

