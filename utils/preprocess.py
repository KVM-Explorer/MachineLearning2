import datetime
from cmath import rect

import numpy as np
import numpy
import torch
import cv2
import tkinter
import numpy as np
from matplotlib import pyplot as plt
starttime = datetime.datetime.now()
from matplotlib import pyplot as plt

print("开始执行")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import os
import cv2


def get_image(X, Y):  # 获取数据集中的图片并保存在数组中
    cnt = 0
    for i in range(0, 3):
        # 遍历文件夹
        for f in os.listdir("./photo/%s" % i):  # 打开文件夹，读取图片
            cnt += 1  # 统计数据集图片数量
            X.append("photo//" + str(i) + "//" + str(f))  # 图片放入列表
            Y.append(i)
    # 将列表转为数组，便于操作
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def dataset_split(X, Y):
    global X0_train, X0_test
    # 将数据集拆分为训练集和测试集，测试集比例为0.3，采用随机拆分，每一次测试集和训练集均不同
    X0_train, X0_test = train_test_split(X, test_size=0.3)

    return X0_train, X0_test

# 对要训练的图片进行数字图像处理，提取特征
def areaCal(contour):
    for i in range(0, len(contour)):
        cnt = 0
        for j in range(0, len(contour)):
            if cv2.contourArea(contour[i]) >= cv2.contourArea(contour[j]):
                cnt = cnt + 1
                if cnt == len(contour):
                    return i

def get_features(array):
    # 拿到数组的高度和宽度
    h, w = array.shape
    data = []
    for x in range(0, w / 4):
        offset_y = x * 4
        temp = []
        for y in range(0, h / 4):
            offset_x = y * 4
            # 统计每个区域的1的值
            temp.append(sum(sum(array[0 + offset_y:4 + offset_y, 0 + offset_x:4 + offset_x])))
        data.append(temp)
    return np.asarray(data)

def trainpic_process(X0_train):
    y0_train=[]
    n=[]
    for i in X0_train:
        # 读取图像
        image = cv2.imread(i)
        max_width = 270
        max_height = 72
        # 图像像素大小一致
        image = cv2.resize(image, (max_width, max_height),
                          interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_height = hsv.shape[0]
        hsv_width = hsv.shape[1]
        for row in range(hsv_height):
            for col in range(hsv_width):
                if hsv[row,col][0]>=28 and hsv[row,col][0] <=77: hsv[row, col] = [0,0,255]
        image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # 计算图像直方图并存储至X数组，图片特征为H 色调值（0，180），S饱和度（0，256）
        hist1 = cv2.calcHist([hsv], [0], None,
                             [18], [1.0, 180.0, ])
        hist2 = cv2.calcHist([hsv], [1], None,
                             [26], [1.0, 255.0])
        # X_train.append(hist.flatten())
        XCC = hist1.flatten() / sum(hist1.flatten())
        XDD = hist2.flatten() / sum(hist2.flatten())
        XCC = np.concatenate((XCC, XDD), axis=None)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_width = gray.shape[1]
        gray_height = gray.shape[0]
        for row in range(gray_height) :
            for col in range(gray_width) :
                if gray[row,col] == 255: gray[row,col] = 0
        k = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
        ret, binary = cv2.threshold(gray,54 , 255, cv2.THRESH_BINARY)

        # cv2.imshow("bin Image",binary)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image = cv2.drawContours(image.copy(), contours, areaCal(contours), (0, 0, 255), 3)

        index = areaCal(contours)
        if index == None: continue
        cnt = contours[index]
        connect_area = cv2.connectedComponents(binary)[0]
        x, y, w, h = cv2.boundingRect(cnt)
        # Straight Bounding Rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Rotate Rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        area = cv2.contourArea(cnt)


        image = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        #　cv2.imshow('img', image)
        # cv2.waitKey(0)
        # X_train.append(rect[1][0])
        # X_train.append(rect[1][1])
        if rect[1][0] > rect[1][1]:
            XCC = np.append(XCC, area/(max_width*max_height))
            XCC = np.append(XCC, rect[1][1]/rect[1][0])
            XCC = np.append(XCC,connect_area/50)
        else:
            XCC = np.append(XCC, area/(max_width*max_height))
            XCC = np.append(XCC, rect[1][0]/rect[1][1])
            XCC = np.append(XCC,connect_area/50)
        # XCC.append(rect[1][0])
        # XCC.append(rect[1][1])
        X_train.append(XCC)
        n = list(filter(str.isdigit, i))
        y0_train.append(int(n[0]))
    y0_train=np.array(y0_train)
    return X_train,y0_train

def traincic_process(X0_test):
    y0_test=[]
    m=[]
    for i in X0_test:
        # 读取图像
        image = cv2.imread(i)
        max_width = 270
        max_height = 72
        # 图像像素大小一致
        image = cv2.resize(image, (max_width, max_height),
                          interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_height = hsv.shape[0]
        hsv_width = hsv.shape[1]
        for row in range(hsv_height):
            for col in range(hsv_width):
                if hsv[row,col][0]>=28 and hsv[row,col][0] <=77: hsv[row, col] = [0,0,255]
        image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # 计算图像直方图并存储至X数组，图片特征为H 色调值（0，180），S饱和度（0，256）
        hist1 = cv2.calcHist([hsv], [0], None,
                             [18], [1.0, 180.0, ])
        hist2 = cv2.calcHist([hsv], [1], None,
                             [26], [1.0, 255.0])
        # X_train.append(hist.flatten())
        XCC = hist1.flatten() / sum(hist1.flatten())
        XDD = hist2.flatten() / sum(hist2.flatten())
        XCC = np.concatenate((XCC, XDD), axis=None)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_width = gray.shape[1]
        gray_height = gray.shape[0]
        for row in range(gray_height) :
            for col in range(gray_width) :
                if gray[row,col] == 255: gray[row,col] = 0
        k = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
        ret, binary = cv2.threshold(gray,54 , 255, cv2.THRESH_BINARY)

        # cv2.imshow("bin Image",binary)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image = cv2.drawContours(image.copy(), contours, areaCal(contours), (0, 0, 255), 3)

        index = areaCal(contours)
        if index == None : continue
        cnt = contours[index]
        connect_area = cv2.connectedComponents(binary)[0]
        x, y, w, h = cv2.boundingRect(cnt)
        # Straight Bounding Rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Rotate Rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        area = cv2.contourArea(cnt)


        image = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        #　cv2.imshow('img', image)
        # cv2.waitKey(0)
        # X_train.append(rect[1][0])
        # X_train.append(rect[1][1])
        if rect[1][0] > rect[1][1]:
            XCC = np.append(XCC, area/(max_width*max_height))
            XCC = np.append(XCC, rect[1][1]/rect[1][0])
            XCC = np.append(XCC,connect_area/50)
        else:
            XCC = np.append(XCC, area/(max_width*max_height))
            XCC = np.append(XCC, rect[1][0]/rect[1][1])
            XCC = np.append(XCC,connect_area/50)
        # XCC.append(rect[1][0])
        # XCC.append(rect[1][1])
        X_test.append(XCC)
        m=list(filter(str.isdigit,i))
        y0_test.append(int(m[0]))
    y0_test = np.array(y0_test)
    return X_test,y0_test


if __name__=="__main__":
    X = []  # 用于存放图片的图像矩阵
    Y = []
    get_image(X, Y)
    dataset_split(X, Y)

    X_train = []  # 创建空列表存放数据
    X_test = []
    y_train = []
    y_test = []

    (X_train, y_train) = trainpic_process(X0_train)

    (X_test, y_test) = traincic_process(X0_test)

    print(X_train)
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(y_train))
    print(np.shape(y_test))
    print(X0_train)
    print(np.shape(X0_test))
    print(y_train)
    print(y_test)
    numpy.array(X_train)
    numpy.array(X_test)
    feature1 = torch.tensor(X_train)
    torch.save(feature1, "../data/train/vectors.pt")

    feature2 = torch.tensor(y_train)
    torch.save(feature2, "../data/train/labels.pt")

    feature3 = torch.tensor(X_test)
    torch.save(feature3, "../data/test/vectors.pt")

    feature4 = torch.tensor(y_test)
    torch.save(feature4, "data/test/labels")