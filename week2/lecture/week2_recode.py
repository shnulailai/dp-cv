#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img = cv2.imread('lenna1.jpg')
cv2.imshow('lenna',img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[3]:


# Gaussian Kernel Effect
g_img = cv2.GaussianBlur(img,(7,7),5)
cv2.imshow('gaussian_blur_lenna', g_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[4]:


# 图像变得更加模糊，因为范围更大，平均效果更加明显
g_img =cv2.GaussianBlur(img,(17,17),5)
cv2.imshow('gaussian_blur_lenna',g_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[5]:


# 图像更加模糊，因为方差变小了，高斯图像更加尖锐，中心点起的作用更大
g_img = cv2.GaussianBlur(img,(7,7),1)
cv2.imshow('gaussian_blur_lenna',g_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[6]:


# 来看看高斯核
kernel = cv2.getGaussianKernel(7,5)
print(kernel)


# In[7]:


# 为啥一维，因为一维快
# 理论解释，用显示地代码看隐式地高斯和显示地分布高斯地效果
g1_img = cv2.GaussianBlur(img,(7,7),5)
# ori depth, kernelX,kernelY
g2_img = cv2.sepFilter2D(img,-1,kernel,kernel) #当depth=-1时，表示输出图像与原图像有相同的深度
cv2.imshow('g1_blur_lenna', g1_img)
cv2.imshow('g2_blur_lenna', g2_img)
key == cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[8]:


######### other Application ####
# 2nd derivative:laplacian(双边缘效果)
kernel_lap = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)
lap_img = cv2.filter2D(img,-1,kernel = kernel_lap)
cv2.imshow('lap_lenna',lap_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[9]:


# 应用：图像锐化 = edge+ori
# app:sharpen
# 图像+edge = 更加锐利地图像，因为突出边缘
kernel_sharp = np.array([[0,1,0],[1,-3,1],[0,1,0]],np.float32)
lap_img = cv2.filter2D(img,-1,kernel = kernel_lap)
cv2.imshow('sharp_lenna',lap_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[10]:


# 这样不对，因为，周围有4个1，中间是-3，虽然有边缘效果，但是周围得1会使得原kernel有滤波效果，使图像模糊；
# 解决：所以取kernel_lap得相反数，再加上原图像，这样突出了中心像素，效果类似于小方差的高斯，所以
#      可以既有边缘效果，又保留图像清晰度
kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
lap_img = cv2.filter2D(img,-1,kernel=kernel_sharp)
cv2.imshow('sharp_lenna',lap_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[11]:


# 更“凶猛”的边缘效果
# 不仅考虑x-y方向上的梯度，同时考虑了对角线方向上的梯度
kernel_sharp = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32) 
lap_img = cv2.filter2D(img,-1,kernel=kernel_sharp)
cv2.imshow('sharp_lenna',lap_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[12]:


######## Edge #########
# x轴
edgex = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
sharp_img = cv2.filter2D(img, -1, kernel=edgex)
cv2.imshow('edgex_lenna', sharp_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[13]:


# y轴
edgey = np.array([[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]], np.float32)
sharpy_img = cv2.filter2D(img, -1, kernel=edgey)
cv2.imshow('edgex_lenna', sharp_img)
cv2.imshow('edgey_lenna', sharpy_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[14]:


######### 角点 ###########
img = cv2.imread('lenna1.jpg')
img = cv2.resize(img, (640, 480))
img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
print(img_gray)

img_harris = cv2.cornerHarris(img_gray,2,3,0.05)
cv2.imshow('img_harris', img_harris)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# 没法看的原因：1.float类型；2.img_harris本质上是每个pixel对于Harris函数的响应值
# 没法看的价值
print(img_harris)
#显示得更加清楚
img_harris = cv2.dilate(img_harris,None)

thres = 0.05 * np.max(img_harris)
img[img_harris > thres] = [0,0,255]
cv2.imshow('img_harris',img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[ ]:


################# SIFT ##############
img = cv2.imread('lenna1.jpg')
# create sift class
sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT
kp = sift.detect(img,None) # None for mask
# compute SIFT descriptor
kp,des = sift.compute(img,kp)
print(des.shape)
img_sift = cv2.drawKeypoints(img,kp,outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('lenna_sift.jpg',img_sift)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(cv2.__version__)


# In[ ]:





# In[ ]:




