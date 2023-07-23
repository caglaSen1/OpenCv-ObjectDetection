'''
Watershed algorithm: used for segmentation, i.e. to separate different objects in an image.
Any grayscale image can be seen as a topographic surface, where high density refers to peaks and hills and low density refers to valleys.
You start filling each isolated valley (local min) with different colored water (labels).
As the water rises, the water from different valleys will obviously start to merge with different colors depending on the nearby peaks (gradients).
To avoid this, you build barriers where the water merges. You keep filling water and building barriers until all the peaks are flooded.
Then the barriers you build give you the segmentation result.
- Background different from the front objects -
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer img in
coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")

# There is too much noise (details) (indentations, protrusions of the coins, details inside the coin, etc.)
# we should get rid of them - it is enough to see only the perimeter of the coins
# Low Pass Filter (lpf) - blurring - apply median blur
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")

# In the black and white photo, high intensity refers to hills and low intensity to valleys.
# The watershed algorithm assigns a different color label for each valley and this is important for sagmantation.
# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")

# binary threshold (segmentation operation) - with this method the difference between backgrounds(valleys) and coins(hills) becomes more clear
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# Make contour for correcting irregularities
# Find the contours
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # it gives us the external contours
        cv2.drawContours(coin, contours, i, (0, 255, 0), 10)
plt.figure(), plt.imshow(coin), plt.axis("off")

# Didn't get a good result...

# WATERSHED Algorithm:

# Transfer img in
coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")

# lpf: blurring
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")

# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")

# binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# There are bridges between coins (they connect at some points)
# We can make opening to fix this
# opening: erosion and dilate (we can first shrink objects and then expand them)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")

# find the distance between objects
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")

# we need to parse between the front and back image
# to get rid of bridges:
# minimize objects
# bridges have small amplitude-threshold values, centers have large amplitude-threshold values
# if we find the max threshold of dist transform and take 40% of it, we remove the bridges, leaving only small islands
ret, sure_foreground = cv2.threshold(dist_transform, 0.4 * np.max(dist_transform), 255, 0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")

# The water filling process is performed. The connection between the objects was broken.

# Enlarge image for background - Now let's enlarge the coins to find out what the background is.
sure_background = cv2.dilate(opening, kernel, iterations=1)
# Let's change foreground to int:
sure_foreground = np.uint8(sure_foreground)
# Take the difference between the background and foreground - have a clearer picture
unknown = cv2.subtract(sure_background, sure_foreground)
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")

# We need to find the markers to provide input to the Wathershed algorithm - we need to provide the connection between the components.
# Connection
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")
# When the white rings around the coins were made black, the islands, that is, the coins, were fully revealed

# watershed - we'll segment with the watershed algorithm
marker = cv2.watershed(coin, marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")
# now real objects have appeared

# Find the contours and draw a line around them
# Find the contours
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours, i, (255, 0, 0), 2)
plt.figure(), plt.imshow(coin), plt.axis("off")

plt.show()
