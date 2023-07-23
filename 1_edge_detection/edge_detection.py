'''
The purpose of object detection is not to detect what the object is.
The aim is to find the width and height of the coordinates of the objects in the image.
Edge detection - A method that aims to identify points where the image brightness changes sharply
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer image in
img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# Detect edges 0-255
img_edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.figure(), plt.imshow(img_edges, cmap="gray"), plt.axis("off")


# Detect edges with median
med_val = np.median(img)
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))  # 67% - lower bound
high = int(min(255, (1 + 0.33)*med_val))  # upper bound

print(low)
print(high)

img_median = cv2.Canny(image=img, threshold1=low, threshold2=high)
plt.figure(), plt.imshow(img_median, cmap="gray"), plt.axis("off")


# Detect edges with median on blurry img
blurred_img = cv2.blur(img, ksize=(5, 5))
plt.figure(), plt.imshow(blurred_img, cmap="gray"), plt.axis("off")

med_val = np.median(blurred_img)
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))  # 67% - lower bound
high = int(min(255, (1 + 0.33)*med_val))  # upper bound

print(low)
print(high)

img_median = cv2.Canny(image=blurred_img, threshold1=low, threshold2=high)
plt.figure(), plt.imshow(img_median, cmap="gray"), plt.axis("off")

plt.show()



