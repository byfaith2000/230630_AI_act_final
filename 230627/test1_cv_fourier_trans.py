import cv2
import numpy as np
from matplotlib import pyplot as plt


def fourier_transform():
    img = cv2.imread('idol.jpg', 0)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    magnitude_spectrum2 = 20*np.log(np.abs(fshift))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    fig = plt.figure()
    fig.canvas.set_window_title('JetsonNano_Fourier')
    plt.subplot(221), plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(magnitude_spectrum2, cmap = 'gray')
    plt.title('Magnitude Spectrum2'), plt.xticks([]), plt.yticks([])

    plt.show()


fourier_transform()