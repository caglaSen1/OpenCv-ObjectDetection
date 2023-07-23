'''
In image processing, point feature matching is an effective method for detecting a specified target in a complex scene.
What we call a feature: Edge, corner, etc.
This method detects single objects instead of multiple objects.
For example, using this method, one can recognize a specific person on a cluttered image, but not any other person.

The Brute-Force matcher matches the identifier of a feature in one image against all other features of another image and returns the match by distance.
It is slow because it checks for matching against all features.

Scale-invariant feature transformation, keypoints are first extracted from a set of reference images and stored.
An object is recognized in a new image by comparing each feature in the new image individually with this stored data and finding candidate matching features based on the Euclidean distance of the feature vectors.
'''

import cv2
import matplotlib.pyplot as plt

chos = cv2.imread("chocolates.jpg", 0)
plt.figure(), plt.imshow(chos, cmap="gray"), plt.axis("off")

cho = cv2.imread("nestle.jpg", 0)
plt.figure(), plt.imshow(chos, cmap="gray"), plt.axis("off")

# orb, sift: these are the methods that reveal the features. We do the matching with a Brute-force matcher.

# orb descriptor - will detect key points between the image and the object we are looking for (features of the object such as corners, edges, etc.)
orb = cv2.ORB_create()

# key point detection - kp: key points, des: destination
kp1, des1 = orb.detectAndCompute(cho, None)  # could have written mask instead of none
kp2, des2 = orb.detectAndCompute(chos, None)

# We will do the matching with Brute-force matcher:
# Let's define Brute Force
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Match the dots
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# visualize matching pictures
plt.figure()
img_matched = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags=2)
plt.imshow(img_matched), plt.axis("off")

# Didn't get a good result. We'll use a sift identifier. Sift is a bit better than orb.
# Sift identifier: It is not important even if the sizes and rotations are different.
sift = cv2.SIFT_create()

# key point detection with sift
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

# Do the matching again with brute force - shift and orb are the methods that reveal the properties
bf = cv2.BFMatcher()

# Match the dots (knnMatch)
matches = bf.knnMatch(des1, des2, k=2)  # knn - key nearest neighbor. First column is the best match and the second column is the second best match and so on...

best_matches = []

for match1, match2 in matches:
    if match1.distance < 0.75*match2.distance:
        best_matches.append([match1])

plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, best_matches, None, flags=2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")
plt.show()










