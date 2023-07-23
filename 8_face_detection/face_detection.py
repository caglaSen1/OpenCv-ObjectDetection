'''
There is an algorithm that we use to do object detection and it trains a lot of positive and negative images and then it is used to detect objects in other images.
For example, if we want to do face recognition, images containing faces are called positive images and images without faces are called negative images.
Now we will use a pre-trained object detection algorithm.
Features:
Edge features - detecting edges
Line features - detecting lines
Four-rectangle features - Detects rectangular interior divided into 4 parts (like a 2x2 square on a chessboard)
On the human face, the eyes can be perceived as a horizontal line, the nose as a vertical line, the mouth as a horizontal wide line.
After training on positive human images, we can have features that can reveal distinctive features such as eyes, nose and mouth.

We will use the fronttalface default from github.com/opencv/opencv/tree/master/data/haarcascades
'''

import cv2
import matplotlib.pyplot as plt

# Einstein example:

einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

# classifier - cascades are pre-trained classifiers - classifies whether a face is present or not (obtained by training with positive and negative images)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# It starts from the upper left and looks with certain scale of rect, if there is no face in the region it looks at, it classifies it as negative and continues by sliding, if necessary, it enlarges the scale of the rectangle it searches.
# After detecting the face, assign it to a rectangle
face_rect = face_cascade.detectMultiScale(einstein)  # make a detection using the classifier inside the einstein image and assign it into a rect

'''
The detectMultiScale function is used to detect objects in a given image. It is a cascade detection algorithm specially 
trained for face detection. The working principle of this algorithm is as follows: The image is scanned for objects using 
many boxes of different sizes.  At each box size, a filter is applied through which every pixel in the image passes and 
the presence of an object is evaluated using a certain threshold value.

The minNeighbors parameter is used to filter out regions that the algorithm identifies as faces.  This parameter specifies 
how many neighboring rectangular regions should surround each detected face candidate.  These neighboring regions are used 
to verify the actual face regions. For example, if minNeighbors=3, a face candidate requires at least three neighboring 
rectangular regions. This helps to filter out false positives (i.e. incorrectly detected faces) and provides more reliable 
face detection.

In general, an increase in minNeighbors means that more neighborhoods are required, resulting in a more precise 
detection, but it also increases the likelihood of less detected faces. On the other hand, decreasing the minNeighbors 
value provides a more liberal detection and detects more face candidates, but the number of false positives may increase.
'''


for (x, y, w, h) in face_rect:
    # After detecting the face draw a rect on the main image to see it
    cv2.rectangle(einstein, (x, y), (x+w, y+h), (255, 255, 255), 10)

plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")



# Barcelona example:

barcelona = cv2.imread("barcelona.jpg", 0)  # siyah beyaz olarak içeriye aktardık
plt.figure(), plt.imshow(barcelona, cmap="gray"), plt.axis("off")

# To avoid inaccuracies we have to play with some parameters of detectMultiScale
face_rect = face_cascade.detectMultiScale(barcelona, minNeighbors=7)
# minNeighbors=3 -> states that the algorithm used for face detection requires at least three neighboring regions
# for each detected face candidate.

for (x, y, w, h) in face_rect:
    cv2.rectangle(barcelona, (x, y), (x+w, y+h), (255, 255, 255), 10)

plt.figure(), plt.imshow(barcelona, cmap="gray"), plt.axis("off")



# example of face detection on video

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors=7)
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 10)

        cv2.imshow("face detect", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.show()
