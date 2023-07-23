import cv2
import os

files = os.listdir()
#print(files)

# make a list of cat pictures
img_path_list = []
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
print(img_path_list)

for j in img_path_list:
    img = cv2.imread(j)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use fronttalface from github.com/opencv/opencv/tree/master/data/haarcascades
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

    rects = detector.detectMultiScale(gray, scaleFactor=1.045, minNeighbors=2)
    # scaleFactor=1 determines how much to zoom in on the image

    for (i, (x, y, w, h)) in enumerate(rects):  # ((x, y, w, h)) - tuple
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(img, "Kedi {}".format(i + 1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow(j, img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue





