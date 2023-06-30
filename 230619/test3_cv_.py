import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_affine_transformation():
    img = cv2.imread('idol.jpg', cv2.IMREAD_COLOR)
    rows, cols = img.shape[:2]

    pts1 = np.float32([[140, 300], [360, 350], [140, 100]])
    pts2 = np.float32([[140, 280], [360, 170], [140, 480]])

    cv2.circle(img, (140, 300), 10, (255, 0, 0), -1)
    cv2.circle(img, (360, 350), 10, (0, 255, 0), -1)
    cv2.circle(img, (140, 100), 10, (0, 0, 225), -1)

    cv2.circle(img, (140, 280), 10, (255, 0, 0), -1)
    cv2.circle(img, (360, 170), 10, (0, 255, 0), -1)
    cv2.circle(img, (140, 480), 10, (0, 0, 225), -1)

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    fig=plt.figure()
    fig.canvas.set_window_title("JetsonNano_Affine_transform")
    plt.subplot(121), plt.imshow(img), plt.title('IMAGE')
    plt.subplot(122), plt.imshow(dst), plt.title('AFFINE')
    plt.show()


image_affine_transformation()