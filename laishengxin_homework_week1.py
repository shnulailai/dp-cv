#!/usr/bin/env python
# coding: utf-8

# Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.
# Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.

# In[65]:


import cv2
import random
import numpy as np
import glob as gb
import math
import time


# In[66]:


# image crop 图片的剪裁
# 把图片裁剪成为圆形背景图像并保存
def img_crop(img):
    # 读取透明通道
#     img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    # 裁剪坐标为[y0:y1, x0:x1],缩小图像比例
#     img = img[0:300,0:300]
    rows,cols,channel = img.shape 
#     print(rows,cols,channel)
    # 创建一张4通道的新图片，包含透明通道，初始化是透明的
    img_crop = np.zeros((rows,cols,4),np.uint8)
    img_crop[:,:,0:3] = img[:,:,0:3]
    
     # 创建一张单通道的图片，设置最大内接圆为不透明，注意圆心的坐标设置，cols是x坐标，rows是y坐标
    img_circle = np.zeros((rows,cols,1),np.uint8)
    img_circle[:,:,:] = 0  # 设置为全透明
    img_circle = cv2.circle(img_circle,(cols//2,rows//2),int(min(rows, cols)/2),(255),-1) # 设置最大内接圆为不透明
    # 图片融合
    img_crop[:,:,3] = img_circle[:,:,0]
    # 保存图片
#     cv2.imwrite("./kobe/kb11_img_crop.png", img_crop)
    return img_crop


# In[67]:


# color shift 图片上色
def color_shift(img):
#     img = cv2.imread(input_img)
     # brightness
    B, G, R = cv2.split(img)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    img_color_shift = cv2.merge((B, G, R))
    # 保存图片
#     cv2.imwrite("./kobe/kb1_color_shift.png", img_color_shift)
    return img_color_shift


# In[68]:


def img_rotation(img, angle, scale=1.):
#     img = cv2.imread(input_img)
    w = img.shape[1]
    h = img.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    img_rotation = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
#     cv2.imwrite("./kobe/kb2_img_rotation.png", img_rotation)
    return img_rotation


# In[69]:


# perspective（投影） transform
def random_warp(img):
#     img = cv2.imread(input_img)
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
#     cv2.imwrite("./kobe/kb3_img_warp.png", img_warp)
    return img_warp


# In[70]:


# 主函数调用
if __name__ == "__main__":
    # 输入文件夹所在的路径
    img_path = gb.glob('./kobe/*')
#     img_crop = img_crop('./kobe/kb1.jpg')
    # 设置角度 
    angle = 45
    count = 1
    for path in img_path:
        # 把所有的反斜杠改为正斜杠
        path = str(path.replace('\\','/'))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # 把图片缩放四分之一
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        if count == 1:
            # 只对第一张图片进行图片裁剪
            img_crop = img_crop(img)
            #  路径拼接
            cv2.imwrite(str(path[0:-3]+'png'), img_crop)
        count = count +  1
        if 1< count < 5 :
            # 给部分图片上色
            color = color_shift(img)
            cv2.imwrite(str(path[0:-3]+'png'), color)
        if 5< count < 7 :
            # irotation 旋转,输入旋转度数
            rotation = img_rotation(img,angle)
            cv2.imwrite(str(path[0:-3]+'png'), rotation)
        if count > 7:
            # perspective(投影) transform
            img_warp = random_warp(img)
            cv2.imwrite(str(path[0:-3]+'png'), img_warp)    

