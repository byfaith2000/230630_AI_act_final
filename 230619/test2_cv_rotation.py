import cv2
import numpy as np
from matplotlib import patches

def image_rotation():
    img = cv2.imread('idol.jpg', cv2.IMREAD_COLOR)
    height, width, c = img.shape

    img = cv2.rectangle(img, (0,0), (width-1, height-1), (255, 255, 255), 1)
    rows, cols = img.shape[:2]

    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 0.5)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('JetsonNano_Original', img)
    cv2.imshow('JetsonNano_Rotation', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_rotation()
