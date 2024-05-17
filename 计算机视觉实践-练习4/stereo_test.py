import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右图像
img_left = cv2.imread('imgs.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('imgss.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if img_left is None or img_right is None:
    print("Error: Failed to read images")
else:
    # 创建StereoBM对象
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # 或者使用StereoSGBM对象
    # stereo = cv2.StereoSGBM_create(minDisparity=0,
    #                                numDisparities=16,
    #                                blockSize=3,
    #                                P1=8 * 3 * 3**2,
    #                                P2=32 * 3 * 3**2,
    #                                disp12MaxDiff=1,
    #                                uniquenessRatio=15,
    #                                speckleWindowSize=100,
    #                                speckleRange=32)

    # 计算视差图
    disparity = stereo.compute(img_left, img_right)

    # 归一化视差图以便显示
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # 使用matplotlib显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Left Image')
    plt.imshow(img_left, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Disparity Map')
    plt.imshow(disparity_normalized, cmap='jet')
    plt.colorbar()
    plt.show()
