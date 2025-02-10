# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:58:09 2021

@author: kini5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
import random

from utils import read_img_mono, display_img, save_img


img = read_img_mono("res/highway.jpg")
# Gaussian blur helps get rid of some noise before edge detection.
img_blur = gaussian_filter(img, sigma=2)

My = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
dMy = convolve2d(img_blur, My, mode="same")
display_img(dMy)
save_img(dMy,'original roadmap.png')

# Heuristic edge detection. You may ingore this part.
edges = dMy.copy()
edges[edges < 110] = 0
display_img(edges)
save_img(edges,'edge detected roadmap.png')
edge_x, edge_y = np.where(edges > 0)
left_edge_x = []
left_edge_y = []
right_edge_x = []
right_edge_y = []
for idx in range(len(edge_x)):
    if edge_x[idx] > 0:
        if edge_y[idx] < 300:
            left_edge_x.append(edge_x[idx])
            left_edge_y.append(edge_y[idx])
        if edge_y[idx] > 500:
            right_edge_x.append(edge_x[idx])
            right_edge_y.append(edge_y[idx])

# Here we have x and y coordinates stored in right_edge_x, right_edge_y,
# left_edge_x, left_edge_y, for the right and left sides of the road
# respectively. Use them to fit a linear boundary in each side of the road.


""" BEGIN SOLUTION """

# left_m, left_b = np.polyfit(left_edge_x, left_edge_y, 1)
# right_m, right_b = np.polyfit(right_edge_x, right_edge_y, 1)

# x_vals = np.array(range(min(left_edge_x), max(left_edge_x)))
# print(x_vals)
# y_vals = left_m * x_vals + left_b
# plt.plot(y_vals, x_vals, 'r-', label="Left Road Boundary")

# x_vals = np.array(range(min(right_edge_x), max(right_edge_x)))
# y_vals = right_m * x_vals + right_b
# plt.plot(y_vals, x_vals, 'b-', label="Right Road Boundary")

# plt.imshow(img, cmap='gray')
# plt.legend()
# plt.show()


# RANSAC Line Fitting Function
def ransac_line_fitting(x, y, num_iterations=100, distance_threshold=5):
    best_m, best_b = None, None
    max_inliers = 0

    x = np.array(x)
    y = np.array(y)

    for _ in range(num_iterations):
        # Randomly select two points
        sample_indices = random.sample(range(len(x)), 2)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]

        # Fit a line (y = mx + b) using these two points
        if x_sample[0] == x_sample[1]:  # Avoid division by zero
            continue

        m, b = np.polyfit(x_sample, y_sample, 1)

        # Compute distances to all points
        distances = np.abs(y - (m * x + b)) / np.sqrt(m**2 + 1)

        # Count inliers within threshold
        inlier_indices = np.where(distances < distance_threshold)[0]
        num_inliers = len(inlier_indices)

        # Update best model if more inliers are found
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_m, best_b = m, b
            best_inliers_x = x[inlier_indices]
            best_inliers_y = y[inlier_indices]

    # Refit using all inliers
    if max_inliers > 2:
        best_m, best_b = np.polyfit(best_inliers_x, best_inliers_y, 1)

    return best_m, best_b

# Apply RANSAC for both left and right edge lines
left_m_ransac, left_b_ransac = ransac_line_fitting(left_edge_x, left_edge_y)
right_m_ransac, right_b_ransac = ransac_line_fitting(right_edge_x, right_edge_y)

x_vals = np.array(range(min(left_edge_x), max(left_edge_x)))
y_vals = left_m_ransac * x_vals + left_b_ransac
plt.plot(y_vals, x_vals, 'r-', label="Left Road Boundary")

x_vals = np.array(range(min(right_edge_x), max(right_edge_x)))
y_vals = right_m_ransac * x_vals + right_b_ransac
plt.plot(y_vals, x_vals, 'b-', label="Right Road Boundary")

plt.imshow(img, cmap='gray')
plt.legend()
plt.show()

'''
RANSAC画出来的图确实比Least Square更加贴近原图中的行道线
'''


""" END SOLUTION """
