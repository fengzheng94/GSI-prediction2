import cv2
import numpy as np

# 读取图像
img = cv2.imread('meitian3.png', cv2.IMREAD_GRAYSCALE)

# 进行Canny边缘检测
edges = cv2.Canny(img, 100, 200)

# 显示原始图像和Canny边缘检测结果
cv2.imshow('Original Image', img)
cv2.imshow('Canny Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()