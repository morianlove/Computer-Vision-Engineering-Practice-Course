import cv2
import numpy as np

# 读取两张图片
img1 = cv2.imread('imgs.jpg')
img2 = cv2.imread('imgss.jpg')

# 转换为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 使用SIFT特征点检测和匹配
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher创建
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 选择最佳匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 获取匹配的关键点
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 计算单应性变换矩阵
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 输出单应性变换矩阵
print("单应性变换矩阵 H:")
print(H)
