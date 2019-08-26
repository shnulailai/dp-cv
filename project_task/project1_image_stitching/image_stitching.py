# -*- coding: utf-8 -*-
# @Time    : 2019/8/26 8:57
# @Author  : laisx
# @File    : image_stitching.py
# 参考一篇特别棒的博文：https://www.jianshu.com/p/f1b97dacc501
'''
Pipline:
1.Find feature points in each image
2.Use RANSAC to find keypoint mathces
3.Use Homography matrix to get transferring info
4.Merge two images
'''
import cv2
import numpy as np

def image_operation():
    img1, img2 = cv2.imread('image_c.jpg'), cv2.imread('image_d.jpg')
    # 修改图像的大小
    img1 = cv2.resize(img1, (600, 600), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (600, 600), interpolation=cv2.INTER_CUBIC)

    # step1:Find feature points in each image by SIFT
    kp1, des1 = get_feature_keypoint(img1)
    kp2, des2 = get_feature_keypoint(img2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    '''
    展示相连匹配关键点
    '''
    # Need to draw only good matches, so create a mask
    # matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.5 * n.distance:
    #         matchesMask[i] = [1, 0]
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=0)
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # cv2.imshow("img3", img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 提取优秀的特征点
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    src_pts = np.array([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])

    # step3:Use Homography matrix to get transferring info
    H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # step4:Merge two images
    h, w = img1.shape[:2]
    shft = np.array([[1.0, 0, w], [0, 1.0, 0], [0, 0, 1.0]])
    M = np.dot(shft, H[0])  # 获取左边图像到右边图像的投影映射关系
    dst_corners = cv2.warpPerspective(img1, M, (w * 2, h))  # 透视变换，新图像可容纳完整的两幅图
    # cv2.imshow('tiledImg1', dst_corners)  # 显示，第一幅图已在标准位置
    dst_corners[0:h, w:w * 2] = img2  # 将第二幅图放在右侧
    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)
    cv2.imshow('result', dst_corners)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_feature_keypoint(img):
    ########### SIFT ###########
    # create sift class
    sift = cv2.xfeatures2d.SIFT_create()
    # detect SIFT
    kp = sift.detect(img, None)  # None for mask
    # compute SIFT descriptor
    kp, des = sift.compute(img, kp)
    # print(des.shape)
    # img_sift = cv2.drawKeypoints(img, kp, outImage=np.array(
    #     []), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des

def main():
    # 输入图像
    image_operation()

def test1():
    # 最简单粗暴的方式，直接利用stitch函数进行拼接
    img1 = cv2.imread('image_c.jpg')
    img2 = cv2.imread('image_d.jpg')
    stitcher = cv2.createStitcher(False)
    # stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA), 根据不同的OpenCV版本来调用
    (_result, pano) = stitcher.stitch((img1, img2))
    cv2.imshow('pano', pano)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
    # 实在解决不了的时候，采取以下的方式，暴力解决
    # test1()
