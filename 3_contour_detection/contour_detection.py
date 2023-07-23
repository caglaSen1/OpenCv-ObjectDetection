import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer img
img = cv2.imread("contour.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# Detect Contour
# contours = cordinates of contours
contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

for i in range(len(contours)):

    # External
    if hierarch[0][i][3] == -1:  # it gives us the external contours
        cv2.drawContours(external_contour, contours, i, 255, -1)  # # 255 color, -1 fill it
    # Internal
    else:
        cv2.drawContours(internal_contour, contours, i, 255, -1)

plt.figure(), plt.imshow(external_contour, cmap="gray"), plt.axis("off"), plt.title("External Contour")
plt.figure(), plt.imshow(internal_contour, cmap="gray"), plt.axis("off"), plt.title("Internal Contour")
plt.show()
