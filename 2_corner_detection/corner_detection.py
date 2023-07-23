'''
Corner detection
Imagine a box at position x, y. If the intensity difference between the current position of the box and the next position is large, we have gone around the corner.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer image in
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# Harris corner detection
dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
plt.figure(), plt.imshow(dst, cmap="gray"), plt.axis("off"), plt.title("Harris corner detection")

# Dilation - increase white areas
dst = cv2.dilate(dst, None)
img[dst > 0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap="gray"), plt.axis("off"), plt.title("Dilation")

# Shi Tomasi detection
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)  # how many corners = 120, quality level = 0.01, min dist = 10
corners = np.int64(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, (125, 125, 125), cv2.FILLED)

plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("Shi Tomasi detection")

plt.show()

