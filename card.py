# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/11 10:59
@version: 1.0
@File: card.py
'''


import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN
import os


def main():
    card_list=['card.png','card2.png','card3.png']
    model_path='model.png'
    # 数字模板
    digits=model(model_path)
    for card_path in card_list:
        # 读取并处理信用卡
        locs,card_gray,card_copy=read(card_path)
        # 匹配并判断
        # cv_show(card_path+'card_resize',card_copy.copy())
        # cv_show(card_path+'card_M',Mmatch(locs, card_gray, digits, card_copy.copy()))
        # cv_show(card_path+'card_EMNIST',Tmatch(locs, card_gray, card_copy.copy()))
        cv2.imshow(card_path+'_resize', card_copy.copy())
        cv2.imshow(card_path+'_M', Mmatch(locs, card_gray, digits, card_copy.copy()))
        cv2.imshow(card_path+'_EMNIST', Tmatch(locs, card_gray, card_copy.copy()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def model(model_path):
    img = cv2.imread(model_path)  # 读取模板图片
    #cv_show('img', img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv_show('img_gray', img_gray)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv_show('img_bin', img_bin)
    ref_contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, ref_contours, -1, (0, 0, 255), 3)  # -1：所有轮廓
    #cv_show('img_C', img)
    #print(np.array(ref_contours, dtype='object').shape)
    #10：表示数字的轮廓
    #dtype = 'object',去警告
    #对轮廓按照数字大小进行排序 方便后面使用 排序思路: 根据每个数字的最大外接矩形的x轴坐标进行排序
    bounding_boxes = [cv2.boundingRect(c) for c in ref_contours]  # 计算每个轮廓的外接矩形
    (ref_contours, bounding_boxes) = zip(*sorted(zip(ref_contours,bounding_boxes), key=lambda b: b[1][0]))
    # print(bounding_boxes) # 这一步的顺序是小到大因为计算方法是右到左
    # print(sorted(bounding_boxes, key=lambda b: b[0]))
    # *：拆开 zip：一一对应且合并要把排序之后的外接矩形和轮廓建立对应关系
    digits = {}
    for (i, c) in enumerate(ref_contours):
        (x, y, w, h) = cv2.boundingRect(c)  # 重新计算外接矩形
        roi = img_bin[y:y + h, x: x + w]  # 取出每个数字 roi：region of interest：感兴趣区域
        roi = cv2.resize(roi, (57, 88))  # resize：合成合适的大小
        digits[i] = roi
    return digits


def read(card_path):
    # 读取输入图像，预处理
    card = cv2.imread(card_path)
    #cv_show('card', card)
    # 信用卡图重构大小，并转换为灰度图
    h, w = card.shape[:2]  # 为保证原图不拉伸需要计算出原图的长宽比
    width = 300
    r = width / w
    card = cv2.resize(card, (300, int(h * r)))
    card_copy=card.copy()
    #cv_show('card_resize', card)
    # 信用卡灰度化、卷积核、形态学顶帽
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    #cv_show('card_gray', card_gray)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    card_tophat = cv2.morphologyEx(card_gray, cv2.MORPH_TOPHAT, rect_kernel)  # 顶帽突出更明亮的区域
    #cv_show('card_tophat', card_tophat)
    # Sobel算子边缘检测
    grad_x = cv2.Sobel(card_tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  #找出边沿
    # print(grad_x)
    grad_x = np.absolute(grad_x)  # 对grad_x进行处理 只用x轴方向的梯度
    min_val, max_val = np.min(grad_x), np.max(grad_x)  # 再把grad_x变成0到255之间的整数
    grad_x = ((grad_x - min_val) / (max_val - min_val)) * 255
    grad_x = grad_x.astype('uint8')  # 修改一下数据类型
    # 闭操作（先膨胀再腐蚀）+二值化 + 闭操作
    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE,rect_kernel)  # 先膨胀再腐蚀可把数字连在一起.
    #cv_show('gradx', grad_x)
    # 通过大津(OTSU)算法找到合适的阈值, 进行全局二值化操作.
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv_show('thresh', thresh)
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  #中间还有空洞再来一个闭操作
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
    #cv_show('thresh', thresh)
    # 计算并画出轮廓cv2.findContours()、cv2.drawContours()
    # 计算轮廓
    thresh_contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(card, thresh_contours, -1, (0, 0, 255), 3)
    #cv_show('img_C', card)
    # 遍历轮廓计算外接矩形, 然后根据实际信用卡数字区域的长宽比, 找到真正的数字区域
    locs = []
    for c in thresh_contours:
        (x, y, w, h) = cv2.boundingRect(c)  # 计算外接矩形
        ar = w / float(h)  # 计算外接矩形的长宽比例
        # print(ar)
        if ar > 2.5 and ar < 4.0:  # 选择合适的区域
            if (w > 40 and w < 55) and (h > 10 and h < 20):  # 在根据实际的长宽做进一步的筛选
                locs.append((x, y, w, h))  # 符合条件的外接矩形留下来
                sorted(locs, key=lambda x: x[0])  # 对符合要求的轮廓进行从左到右的排序.
    return locs,card_gray,card_copy


def Mmatch(locs,card_gray,digits,card):
    output = []
    for (i, (gx, gy, gw, gh)) in enumerate(locs):  # 遍历每一个外接矩形, 通过外接矩形可以把原图中的数字抠出来
        group = card_gray[gy - 5: gy + gh + 5, gx - 5: gx + gw + 5]  # 抠出数字区域, 并且加点余量
        #cv_show('group', group)
        # 对取出灰色group做全局二值化处理
        _,group_bin = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv_show('group_bin', group_bin)
        # 轮廓计算、排序、尺寸resize
        digit_contours, _ = cv2.findContours(group_bin, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in digit_contours]
        (digit_contours, _) = zip(*sorted(zip(digit_contours, bounding_boxes),key=lambda b: b[1][0]))  # 轮廓排序
        group_output = []  # 定义每一组匹配到的数字的存放列表
        for c in digit_contours:  # 遍历排好序的轮廓
            (x, y, w, h) = cv2.boundingRect(c)  # 找到当前数字的轮廓, resize成合适的大小
            roi = group_bin[y: y + h, x: x + w]  # 取出数字
            roi = cv2.resize(roi, (57, 88))
            #cv_show('roi', roi)
            # 模板匹配
            scores = []  # 定义保存匹配得分的列表
            for (digit, digit_roi) in digits.items():  # items：取出key和值
                result = cv2.matchTemplate(roi, digit_roi, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)  # 只要最大值即分数
                scores.append(score)
            group_output.append(str(np.argmax(scores)))  # 找到分数最高的数字,即匹配到的数字
        # 轮廓绘制
        cv2.rectangle(card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0,0, 255), 1)
        # 数字显示
        cv2.putText(card, ''.join(group_output), (gx, gy - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        output.extend(group_output)
    return card


def cv_show(name,img):
    root = tk.Tk()
    center_x=root.winfo_screenwidth() // 2
    center_y=root.winfo_screenheight() // 2
    if len(img.shape)== 3:
        img_h,img_w,_=img.shape
    else:
        img_h, img_w= img.shape
    t_x, t_y = (center_x - img_w // 2), (center_y - img_h // 2)
    cv2.imshow(name, img)
    cv2.moveWindow(name, t_x, t_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)
    def predict(self, img):
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([0 + img])
        y = self.cnn.model.predict(x,verbose=1)
        return np.argmax(y[0])


def Tmatch(locs,card_gray,card):
    cnn=Predict()
    output = []
    for (i, (gx, gy, gw, gh)) in enumerate(locs):  # 遍历每一个外接矩形, 通过外接矩形可以把原图中的数字抠出来
        group = card_gray[gy - 5: gy + gh + 5, gx - 5: gx + gw + 5]  # 抠出数字区域, 并且加点余量
        # cv_show('group', group)
        # 对取出灰色group做全局二值化处理
        _, group_bin = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv_show('group_bin', group_bin)
        # 轮廓计算、排序、尺寸resize
        digit_contours, _ = cv2.findContours(group_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in digit_contours]
        (digit_contours, _) = zip(*sorted(zip(digit_contours, bounding_boxes), key=lambda b: b[1][0]))  # 轮廓排序
        group_output = []  # 定义每一组匹配到的数字的存放列表
        for c in digit_contours:  # 遍历排好序的轮廓
            (x, y, w, h) = cv2.boundingRect(c)  # 找到当前数字的轮廓, resize成合适的大小
            roi = group_bin[y: y + h, x: x + w]  # 取出数字
            roi = cv2.resize(roi, (28, 28))
            #cv_show('roi', roi)
            # MNIST识别
            group_output.append(str(cnn.predict(roi)))
        # 轮廓绘制
        cv2.rectangle(card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
        # 数字显示
        cv2.putText(card, ''.join(group_output), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        output.extend(group_output)
    return card







if __name__ == '__main__':
    main()
