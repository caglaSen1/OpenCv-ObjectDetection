# Template matching is a method for searching and locating the position of a template image in a larger image.
# It slides the template image over the input image and compares the template and patch of the input image below the template image.

import cv2
import matplotlib.pyplot as plt

# Transfer img
img = cv2.imread("cat.jpg", 0)
print(img.shape)
template = cv2.imread("cat_face.jpg", 0)
print(template.shape)
h, w = template.shape

# Methods for template matching:
# The main goal of these 6 methods provided by openCv is to extract the correlation between two images
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for method in methods:
    method = eval(method)  # eval() - translates from string to function

    matched_result = cv2.matchTemplate(img, template, method)
    print(matched_result.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw on the original image by enclosing the detected area in a rectangle
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.figure()
    plt.subplot(121), plt.imshow(matched_result, cmap="gray"), plt.title("Matched Result"), plt.axis("off")
    # plt.subplot(121) - Say there are 1 row, 2 columns - we use the 1st one
    plt.subplot(122), plt.imshow(img, cmap="gray"), plt.title("Detected Result"), plt.axis("off")
    # plt.subplot(122) - Say there are 1 row, 2 columns - we use the 2nd one

    plt.suptitle(method)
    plt.show()

