import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# from numpy import trace
# from numpy.linalg import det
from scipy.ndimage.filters import gaussian_filter
# from scipy.signal import convolve2d

from utils import read_img_mono, display_img


def detect_corner_points(img, sigma=0.3, alpha=0.06, threshold=40):
    """ This function implements the Harris corner detector.

        It accepts an image as input and returns the x, y coordinates
        of the detected corner points.
    """
    raise NotImplementedError()  # TODO: Implement.


def detect_corner_points_cv(img, num_neighbors=3, sobel_op=3, alpha=0.06):
    """ OpenCV equivalent of detect_corner_points. Use for comparison.
    """
    harris_img = cv.cornerHarris(img, num_neighbors, sobel_op, alpha)
    threshold = 0.01 * harris_img.max()
    x, y = np.where(harris_img > threshold)
    return x, y


def main():
    img = read_img_mono('res/houses.jpg')
    img = gaussian_filter(img, sigma=0.1)
    display_img(img)

    corner_points_x, corner_points_y = detect_corner_points(img)

    im = plt.imread('res/houses.jpg')
    plt.imshow(im)
    plt.scatter(corner_points_y, corner_points_x, s=8, marker='x', c='red')

    plt.show()


if __name__ == '__main__':
    main()
