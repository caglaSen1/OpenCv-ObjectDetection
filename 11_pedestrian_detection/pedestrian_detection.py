import cv2
import os

files = os.listdir()
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)

print(img_path_list)

# hog identifier - detection algorithm
hog = cv2.HOGDescriptor()

# add SVM to identifier
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # called the detector, which is necessary for the detection of people

for img_path in img_path_list:
    print(img_path)
    image = cv2.imread(img_path)

    (rects, weights) = hog.detectMultiScale(image, padding=(8, 8), scale=1.05)
    # padding=(8, 8) fills the spaces around the image with 0 while moving around with an 8x8 window so that do not lose size

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Pedestrian", image)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue
