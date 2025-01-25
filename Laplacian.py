import cv2
import numpy as np

# 读取图像
img = cv2.imread('d46.png', cv2.IMREAD_GRAYSCALE)

# 应用Laplacian算子进行边缘检测
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# 将结果转换为uint8类型
laplacian = np.uint8(np.absolute(laplacian))

# 显示原始图像和Laplacian边缘检测结果
cv2.imshow('Original Image', img)
cv2.imshow('Laplacian Edges', laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()