#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
# 一维卷积
def medianBlur_1(img, kernel, padding_way):
    imarray = np.array(img)
    # 求出需要补0的个数（前后都补的话*2）
    padding = int(kernel/2)
    # 添加数组长度
    n = len(img)+2*padding
    # 根据补0拼接成新的数组
    temp_arr = np.zeros((n,), dtype=np.int)
    if padding_way == 'ZERO': 
        # 将原始数值填充
        temp_arr[padding:-padding] = img
    elif padding_way == 'REPLICA': 
        # 将原始数值填充首末数值
        temp_arr[0:padding] = img[0]
        temp_arr[padding:-padding] = img
        temp_arr[-padding:] = img[len(img)-1]
    else:
        return 
    result_list = []
    # 遍历数组，寻找中位数
    for i in range(n):
        if i == n-padding-1:
            break
        # 每三个遍历
        result_list.append(np.median(temp_arr[i:i+kernel]))
    return result_list


# In[21]:


#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
# 二维卷积
def medianBlur_2(img, kernel, padding_way):
    h,w=img.shape[0],img.shape[1]
    print(img)
    # 确认补充0的宽高
    padding_n,padding_m = int(kernel[0]/2),int(kernel[1]/2)
    # 建立新的图像宽度
    img1 = np.zeros((h+padding_n*2,w+padding_m*2),np.uint8)
    # 将原始数值填充,对矩阵进行扩容，形成新的img1图像
    img1[padding_m:-padding_m,padding_n:-padding_n] = img[:,:]
    if padding_way=="REPLICA":
        # 补齐上边部分
        img1[0:padding_n][:] = img1[padding_n][:]    
        # 补齐下边部分
        img1[-padding_n:][:] = img1[-padding_n-1][:]   
        # 补齐左边部分
        for k in range(padding_m):
            img1[:,k] = img1[:,padding_m] 
        # 补齐右边部分
        for kk in range(-padding_m,-1):
            img1[:,kk] = img1[:,-padding_m-1]
        # range不包含最后一项（-1）
        img1[:,-1] = img1[:,-padding_m-1]
        print(img1)
    elif padding_way=="ZERO":
        pass
    else:
        return     
    # 卷积核从数值为原始数值的地方开始遍历
    # 遍历每一行每一列
    for i in range (padding_n,h-1):
        for j in range (padding_m,w-1):
            # 卷积核 m*n,取出核内的所有值的中位数，确定核的位置            
            img[i,j]=np.median(img1[i-int(padding_n/2):i-int(padding_n/2)+padding_n,j-int(padding_m/2):j-int(padding_m/2)+padding_m])
    return img


# In[24]:


def main():
    # assume 1
    img = [1,3,3,11,2,1,1,23]
    kernel = 3
#   padding_way = 'ZERO'
    padding_way = 'REPLICA'
    result = medianBlur_1(img, kernel, padding_way)
    print(result)
    #  assume 2 单通道处理
    img = cv2.imread('lenna1.jpg',0)
    cv2.imshow("img0",img)
    # 用填充0处理
#     padding_way = 'ZERO'
    # 用边界值填充
    padding_way = 'REPLICA'
    # （3，3） （5，5） （7，7）
    kernel = [9,9]
    img1 = medianBlur_2(img, kernel, padding_way)  
    cv2.imshow("img1",img1)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


# In[25]:


if __name__ == '__main__':
    main()   


# In[ ]:




