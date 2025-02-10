import numpy as np
import time
from scipy.signal import convolve2d

# Define a sample image (a simple grayscale array for testing)
np.random.seed(42)  # For reproducibility
img = np.random.rand(100, 100) * 255  # Random grayscale image

# Define filters
blur_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # 3x3 Gaussian blur kernel
edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel filter (horizontal edges)

# Approach 1: Convolve filters first, then apply to image
start_time = time.time()
combined_filter = convolve2d(edge_filter, blur_filter, mode='full')  # Convolving filters first
print(f'combined_filter is\n {combined_filter}')
result_1 = convolve2d(img, combined_filter, mode='same')  # Apply to image
time_1 = time.time() - start_time

# Approach 2: Apply filters sequentially
start_time = time.time()
blurred_img = convolve2d(img, blur_filter, mode='same')  # First blur the image
result_2 = convolve2d(blurred_img, edge_filter, mode='same')  # Then apply edge detection
time_2 = time.time() - start_time

# Compute numerical difference
difference = np.abs(result_1 - result_2).sum()

# Display results
import ace_tools_open as tools
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format  # 显示 6 位小数
# Compare times
df = pd.DataFrame({
    "Method": ["Convolve Filters First", "Sequential Convolution"],
    "Time (seconds)": [time_1, time_2],
    "Difference Sum": [difference, difference]  # Should be near zero if associativity holds
})

tools.display_dataframe_to_user(name="Associativity of Convolution", dataframe=df)
