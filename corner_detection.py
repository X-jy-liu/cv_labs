import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy import trace
from numpy.linalg import det
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d

from utils import read_img_mono, display_img


def detect_corner_points(img, sigma=0.3, alpha=0.06, threshold=40):
    """ This function implements the Harris corner detector.

        It accepts an image as input and returns the x, y coordinates
        of the detected corner points.
    """
    x = img.shape[0]
    y = img.shape[1]
    harris_img = np.zeros((x, y))
    # Prewitt's kernels.
    Mx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    My = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    # Step 1: Compute horizontal and vertical derivatives.
    Ix = convolve2d(img, Mx, mode='same')
    Iy = convolve2d(img, My, mode='same')
    # Step 2: Compute outer product.
    IxIy = Ix * Iy
    # Step 3: Apply Gaussian filters.
    Ix = gaussian_filter(Ix, sigma)
    Iy = gaussian_filter(Iy, sigma)
    IxIy = gaussian_filter(IxIy, sigma)
    # START DEBUG LINES
    # display_img(Ix)
    # display_img(Iy)
    # display_img(IxIy)
    # END DEBUG LINES
    # Step 4: Compute the Harris score (the slow way).
    for i in range(x):
        for j in range(y):
            H = [
                [np.square(Ix[i, j]), IxIy[i, j]],
                [IxIy[i, j], np.square(Iy[i, j])]
            ]
            harris_score = det(H) - alpha * np.square(trace(H))
            harris_img[i, j] = harris_score
    # Step 5: Apply threshold.
    x, y = np.where(harris_img > threshold)
    return x, y


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
