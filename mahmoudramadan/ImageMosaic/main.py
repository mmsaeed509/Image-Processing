import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
matching_algo = 'knn'
#Here is loading the images by reading them
image_one = cv2.imread('first.jpg')
image_one = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
image_one_gray = cv2.cvtColor(image_one, cv2.COLOR_RGB2GRAY)
#cv2.imshow("Gray Image :", image_one_gray)
image_two = cv2.imread('third.jpg')
image_two = cv2.cvtColor(image_two, cv2.COLOR_BGR2RGB)
image_two_gray = cv2.cvtColor(image_two, cv2.COLOR_RGB2GRAY)
#cv2.imshow("Gray Image :", image_two_gray)
#cv2.waitKey(0)
#End of the loading images

#Here we are displaying the two images in x y directions
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
ax1.imshow(image_one, cmap='gray')
ax1.set_xlabel('Image one', fontsize=14)
ax2.imshow(image_two, cmap='gray')
ax2.set_xlabel('Image two', fontsize=14)
plt.show()
#The End of displaying images

#Here the function to select the descriptor method function
def select_descriptor_method(image, method=None):
    assert method is not None, "Please define a descriptor method. Accepted values are: 'sift', 'surf', 'orb', 'brisk'"
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    if method == 'surf':
        descriptor = cv2.SURF_create()
    if method == 'orb':
        descriptor = cv2.ORB_create()
    if method == 'brisk':
        descriptor = cv2.BRISK_create()
    (key_points, features) = descriptor.detectAndCompute(image, None)
    return key_points, features
#Here is the end of the function that get key points and descriptors

#Here is calling of function select_descriptor method
key_points_image_one, features_image_one = select_descriptor_method(image_one_gray, method='sift')
key_points_image_two, features_image_two = select_descriptor_method(image_two_gray, method='sift')
#End of select function

#Here to get the details of key point of each edge
for keypoint in key_points_image_one:
    x, y = keypoint.pt
    size = keypoint.size
    orientation = keypoint.angle
    response = keypoint.response
    octave = keypoint.octave
    class_id = keypoint.class_id


#Here is the displaying of key points on our images
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(image_one_gray, key_points_image_one, None, color=(255, 0, 0)))
ax1.set_xlabel("Image one key points")
ax2.imshow(cv2.drawKeypoints(image_two_gray, key_points_image_two, None, color=(255, 0, 0)))
ax2.set_xlabel("Image two key points")
plt.show()
#End of the show code

#Here is the function of create matching object to match the edges
def create_match_object(method, crossCheck):
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf
#End of the match function

#Here is the function BF of matching key points
def key_points_matching(feature_image_one, feature_image_two, method):
    bf = create_match_object(method, crossCheck=True)
    regarded_matches = bf.match(feature_image_one, feature_image_two)
    raw_matches = sorted(regarded_matches, key=lambda x: x.distance)
    print("Raw matches with brute force: ", len(raw_matches))
    return raw_matches
#End of the matching using BF matching

#Here is the K-NN edge matching
def key_points_matching_knn(feature_image_one, feature_image_two, ratio, method):
    bf = create_match_object(method, crossCheck=False)
    raw_matches = bf.knnMatch(feature_image_one, feature_image_two, k=2)
    print("Raw matches with KNN: ", len(raw_matches))
    knn_matches = []
    for m, n in raw_matches:
        if m.distance < n.distance * ratio:
            knn_matches.append(m)
    return knn_matches
#End of K-NN matching

#Here is displaying matching edges in the two images in one image
print("Drawing matched features for brute-force algorithm")
fig = plt.figure(figsize=(20, 8))
if matching_algo == 'bf':
    matches = key_points_matching(features_image_one, features_image_two, method='sift')
    matched_featured_image = cv2.drawMatches(image_one, key_points_image_one, image_two, key_points_image_two, matches[:100],
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
elif matching_algo == 'knn':
    matches = key_points_matching_knn(features_image_one, features_image_two, ratio=0.75, method='sift')
    matched_featured_image = cv2.drawMatches(image_one, key_points_image_one, image_two, key_points_image_two,
    np.random.choice(matches, 100), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matched_featured_image)
plt.show()

#Here is the Homography matrix
def fusion_homography(keypoint_img_one, keypoint_img_two, matches, reprojThresh):
    #convert to numpy array
    keypoint_img_one = np.float32([keypoint.pt for keypoint in keypoint_img_one])
    keypoint_img_two = np.float32([keypoint.pt for keypoint in keypoint_img_two])
    if len(matches) > 4:
        points_img_one = np.float32([keypoint_img_one[m.queryIdx] for m in matches])
        points_img_two = np.float32([keypoint_img_two[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(points_img_one, points_img_two, cv2.RANSAC, reprojThresh)
        return matches, H, status
    else:
        return None

M = fusion_homography(key_points_image_one, key_points_image_two, matches, reprojThresh=4)
if M is None:
    print('Error')
else:
    (matches, Homography_Matrix, status) = M

width = image_one.shape[1] + image_two.shape[1]
height = max(image_one.shape[0], image_two.shape[0])
result = cv2.warpPerspective(image_one, Homography_Matrix, [width, height])
result[0:image_two.shape[0], 0:image_two.shape[1]] = image_two
plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(result)
cv2.imwrite("Large_resulted_Image.jpg", result)
plt.show()

