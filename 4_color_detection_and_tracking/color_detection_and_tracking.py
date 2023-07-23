'''
Detect objects in certain colors with the contour detection method

RGB vs HSV
rgb - cube shape; Red, Green, Blue
hsv - Hue, Saturation, Value
'''

import cv2
import numpy as np
from collections import deque  # Use deque to store the center of the detected object

# data type to store the object center
buffer_size = 16
center_points = deque(maxlen=buffer_size)

# blue color range - HSV (hue, saturation, brightness)
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

# capture
cap = cv2.VideoCapture(0)
cap.set(3, 960)  # the width of the camera 960
cap.set(4, 480)  # the height of the camera 480

while True:
    success, frame = cap.read()

    # If we cannot perform the read operation due to the camera, openCV does not give an error and the frame may be
    # empty even though we think we have done it.
    if success:

        # blur the frames and reduce the details, try to eliminate noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # kernel size (11,11), standard deviation is 0 in x and y axises

        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)

        # Create a mask to detect the blue
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("Mask Image", mask)
        # Need to remove the noise around the mask - get rid of the noise by successive erosion and dilate operations.
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("Mask + erode + dilate", mask)  # black and white image - blues look white

        # contour - we can think of it as a geometric shape
        # extraction of contours:
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Used mask copy to avoid breaking the mask
        # Used cv2.RETR_EXTERNAL because if I find the outer contour/part of something blue, I have found the blue thing.

        '''
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE):
        Gets a mask image, this mask image is usually used as a threshold or a mask image used to identify an object. 
        is obtained after segmentation. 
        
        findContours() takes three main parameters:
        image: The input image for contour detection.
        mode: Specifies the contour acquisition mode. The most commonly used mode is cv2.RETR_EXTERNAL, which means that only the outermost contours are retrieved.
        method: Specifies the contour approximation method. cv2.CHAIN_APPROX_SIMPLE is used when the contours are drawn from almost vertical, horizontal or diagonal lines. 
        is used to approximate contours for cases where contours occur.
        
        The findContours() function returns a list of contour points since the function was called. 
        These points are a linear representation of the contour, each representing a (x, y) coordinate pair.
        '''

        center = None  # will be the center of our object - store it in deque

        if len(contours) > 0:  # if able to find a contour, that is, if able to find a blue object.
            # take the largest contour - the largest is always the best
            c = max(contours, key=cv2.contourArea)  # take the maximum of the contours by area

            # turn the contour into a rectangle
            rect = cv2.minAreaRect(c)
            # what is inside this rectangle:
            # x and y coordinates, width, height and rotation of the object on the image

            # need to extract them (rectangle is a tuple)
            ((x, y), (width, height), rotation) = rect

            info = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation))
            print(info)

            # use the rect to get a box - can use this box to enclose the object
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # moment - the image moment is a weighted average of the image pixel intensities from which we can find some
            # properties of an image such as radius, area, center of gravity etc.
            # moment - to find the center of the image
            M = cv2.moments(c)  # send the largest contour into the moment (find the center of gravity of the contour with the largest area)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            # For x = the value at moment 1.0 divided by the value at moment 0.0.
            # For y = the value at moment 0.1 divided by the value at moment 0.0

            # Draw contour: yellow
            cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)  # (0,255,255) - yellow, 2 - thickness

            # draw a dot in the center - pink
            cv2.circle(frame, center, 5, (255, 0, 255), -1)  # 5 - radius

            # print information to screen
            cv2.putText(frame, info, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)  # (25, 50) - coordinates

            # tracking algorithm - to help it remember the past as it detects
            # deque - add dots
            center_points.appendleft(center)
            # Rotate inside the deque to look at the dots and plot them to original image
            for i in range(1, len(center_points)):
                if center_points[i - 1] is None or center_points[i] is None:
                    continue
                cv2.line(frame, center_points[i - 1], center_points[i], (0, 255, 0), 3)

        cv2.imshow("Orijinal Detect", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
