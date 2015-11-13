import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2.cv as cv

# read image frame by inputing a frame number. The returned frame is also scaled by the input scale factor
def getFrame(vidName, frameNum, scale):
    invid   = cv2.VideoCapture(vidName)
    width   = int(invid.get(cv.CV_CAP_PROP_FRAME_WIDTH)*scale)
    height  = int(invid.get(cv.CV_CAP_PROP_FRAME_HEIGHT)*scale)
    fps     = float(invid.get(cv.CV_CAP_PROP_FPS))
    vidLen  = int(invid.get(cv.CV_CAP_PROP_FRAME_COUNT))
    print width, height, fps, vidLen

    frame   = None
    for i in range(vidLen):
        _,img   = invid.read()
        if i == frameNum:
            frame = img
            frame = cv2.resize(frame,(int(width),int(height)))
            break
    del invid
    return frame

# store each frame onto disk
def saveFrames(vidName, scale):
    invid   = cv2.VideoCapture(vidName)
    width   = int(invid.get(cv.CV_CAP_PROP_FRAME_WIDTH)*scale)
    height  = int(invid.get(cv.CV_CAP_PROP_FRAME_HEIGHT)*scale)
    fps     = float(invid.get(cv.CV_CAP_PROP_FPS))
    vidLen  = int(invid.get(cv.CV_CAP_PROP_FRAME_COUNT))
    print width, height, fps, vidLen

    for i in range(vidLen):
        _,img   = invid.read()
        img = cv2.resize(img,(int(width),int(height)))
        if vidName == 'football_left.mov':
            cv2.imwrite('frames/fr_l_{}.jpg'.format(i),img)
        elif vidName == 'football_mid.mov':
            cv2.imwrite('frames/fr_m_{}.jpg'.format(i),img)
        else:
            cv2.imwrite('frames/fr_r_{}.jpg'.format(i),img)
    del invid

#ratio pruning
def pruneMatchesByDistRatio(matches, ratio):
    #each match stored two matches
    #we compute the ratio of the closest match's distance (distance measures how close the matched point is to our query point
    #the smaller the distance, the better the match) to the second closest match's distance. if the closest match is much better
    #than the second closest match, the ratio will be small. otherwise, the ratio will be close to 1.
    #we'll keep matches whose ratio is small.
    ratioPrunedMatches  = [m for m in matches if (m[0].distance*1.0/m[1].distance) < ratio]
    return ratioPrunedMatches

# symetry pruning
def pruneMatchesBySymetry(matches12, matches21):
    #matches12 are matched points from image 1 to image 2
    #matches21 are matched points from image 2 to image 1
    #good match should have consistent pairing. so we'll
    #keep matches whose closest match is always the other point
    #regardless matching was done forward(image 1 to image 2) or backward (image 2 to image1).
    prunedMatches       = []
    for m12 in matches12:
        for m21 in matches21:
            if m12[0].queryIdx == m21[0].trainIdx and m21[0].queryIdx == m12[0].trainIdx:
                prunedMatches.append(m12)
                break
    return prunedMatches

# calculate fundamental matrix of the matched points using ransac
def pruneMatchesRansac(pts1, pts2):
    if len(pts1) != len(pts2):
        print "[ERROR apiFeature.pruneMatchesRansac] input not same length"       
    if len(pts1)>=8:
        #F is the fundamental matrix.
        #this method find F which minimizes the outliers
        #RANSAC is a fast method for ind the F by randomly selecting a few points from the pool for testing instead of
        #naively trying all point pairs from the pool which will be very slow
        F, inliers      = cv2.findFundamentalMat(np.float32(pts1), np.float32(pts2), cv2.cv.CV_FM_RANSAC, 3, 0.99)
        # inliers is a binary array with 0 indicating outliers and 1 indicating inliers. Size of inliers is same as input point
        # convert liners, pts1, pts2's to numpy supported structures so as to use numpy to filter points faster
        inliers         = inliers.ravel()
        pts1            = np.array(pts1)
        pts2            = np.array(pts2)
        #we keep points whose inlier is 1
        pts1            = pts1[inliers==1]
        pts2            = pts2[inliers==1]
    return pts1, pts2

# function to stack images side by side and draw matches. If input pts1 and pts2 are of different sizes, no matches are drawn
def drawMatchedKeypoints(img1, img2, pts1, pts2):
    image1              = img1.copy()
    image2              = img2.copy()
    # conver all to BGR images
    if len(image1.shape)<3:
        image1          = cv2.cvtColor(image1,cv2.COLOR_GRAY2BGR)
    if len(image2.shape)<3:
        image2          = cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)
    # stack images side by side
    h1,w1               = image1.shape[0],image1.shape[1]
    h2,w2               = image2.shape[0],image2.shape[1]
    hmax                = max(h1,h2)
    combinedImg         = np.zeros((hmax,w1+w2,3), dtype='uint8')
    combinedImg[:,:w1]  = image1
    combinedImg[:,w1:]  = image2
    
    if len(pts1) == len(pts2):
        #draw matches
        for i in range(len(pts1)):
            queryPt         = pts1[i]
            trainPt         = pts2[i]
            color           = tuple([np.random.randint(0, 255) for _ in xrange(3)])
            cv2.circle(combinedImg,(int(queryPt[0]),int(queryPt[1])), 2, color,2)
            cv2.circle(combinedImg,(int(trainPt[0]+w1),int(trainPt[1])), 2, color,2)
            cv2.line(combinedImg,(int(queryPt[0]),int(queryPt[1])),(int(trainPt[0]+w1),int(trainPt[1])),color,1)
    return combinedImg

# stitch the two images 
def stitchRight2Mid(img1, img2, pts1, pts2):
    image1              = img1.copy()
    image2              = img2.copy()
    print 'points from first image'
    print pts1
    print 'points from second image'
    print pts2
    #compute homography matrix. Note this matrix just need to be computed once and can be applied to all your video frames afterwards
    H,inliers           = cv2.findHomography(np.float32(pts2), np.float32(pts1), cv.CV_RANSAC)
    image               = cv2.warpPerspective(image2,H,(image1.shape[1]+image2.shape[1],image1.shape[0]))
    image[:,:image1.shape[1],:]=image1
    f = open('homography.right','w')
    for row in H:
        for e in row:
            f.write(str(e) + " ")
    f.close()
    return image

# stitch the two images 
def stitchLeft2Mid(img1, img2, pts1, pts2):
    image1              = img1.copy()
    image2              = img2.copy()
    print 'points from first image'
    print pts1
    print 'points from second image'
    print pts2
    tmp = np.zeros((img1.shape[0], img1.shape[1]*2+500, img1.shape[2]), np.uint8)
    tmp[:,img1.shape[1]+500:,:] = img2
    image2 = tmp
    for i in range(len(pts2)):
        pts2[i] = [pts2[i][0]+img1.shape[1]+500, pts2[i][1]]
    #compute homography matrix. Note this matrix just need to be computed once and can be applied to all your video frames afterwards
    H,inliers           = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv.CV_RANSAC)
    image               = cv2.warpPerspective(image1,H,(image2.shape[1], image2.shape[0]))
    image[:,image1.shape[1]+500:,:] = image2[:,image1.shape[1]+500:,:]
    #write homography into files
    f = open('homography.left','w')
    for row in H:
        for e in row:
            f.write(str(e) + " ")
    f.close()
    return image

def getMatchedPointsFromImages(img1, img2):
    image1              = img1.copy()
    image2              = img2.copy()
    if len(image1.shape)>2:
        image1          = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    if len(image2.shape)>2:
        image2          = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    #init keypoint detector
    detector            = cv2.FeatureDetector_create("SIFT")
    #init keypoint descriptor extractor
    extractor           = cv2.DescriptorExtractor_create("SIFT")
    #init descriptor matcher
    matcher             = cv2.DescriptorMatcher_create("BruteForce")
    #detect keypoints in both images
    keypoints1          = detector.detect(image1, None)
    keypoints2          = detector.detect(image2, None)
    #compute descriptors on detected keypoints
    (kpts1,descriptors1)= extractor.compute(img1,keypoints1)
    (kpts2,descriptors2)= extractor.compute(img2,keypoints2)
    #match descriptors from image1 to image2 and image 2 to image1 so that later we can use symetry pruning later
    #for each match, we find 2 closest matches so that we can use ratio test later
    matches12           = matcher.knnMatch(descriptors1, descriptors2, 2)
    matches21           = matcher.knnMatch(descriptors2, descriptors1, 2)
    # ratio test is to remove ambiguous matches which the closest match and second closest match are very similar
    # this means it is very likely the closest match is wrong or in other words, we cannot decide whether we should
    # use the closets match or the second closest match. Therefore, we discard these ambiguous matches
    matches12RatioTested= pruneMatchesByDistRatio(matches12,0.9)
    matches21RatioTested= pruneMatchesByDistRatio(matches21,0.9)
    # a strong match should give same pairs of points whether we match forward or backward. So we discard points which
    # does not satisfy this symetry criteria 
    matches             = pruneMatchesBySymetry(matches12RatioTested, matches21RatioTested)
    # the next pruning needs coordinates of the matched pairs, so we first store the points into lists
    matchedPts1         = []
    matchedPts2         = []
    for match in matches:
        queryIdx        = match[0].queryIdx
        trainIdx        = match[0].trainIdx
        matchedPts1.append(kpts1[queryIdx].pt)
        matchedPts2.append(kpts2[trainIdx].pt)
    # as a last pruning step, we calculate the fundamental matrix of the matched points and remove outliers
    matchedPts1,matchedPts2 = pruneMatchesRansac(matchedPts1, matchedPts2)
    return matchedPts1, matchedPts2

# mouse event for clicking to manually select matched points.
# you must select matched features in pairs, ie, one point from each picture before moving on to the next match
def mouseEvent(event, x, y, flags, param):
    global image, imageCopy, pts1, pts2, frameX0, frameX1, frameX2
    if event == cv2.EVENT_LBUTTONUP:
        reset()
        # if point in second picture, minus x by first picture width
        if x>=frameX1.shape[1]:
            pts2.append((x-frameX1.shape[1],y))
        else:
            pts1.append((x,y))
        image   = drawMatchedKeypoints(frameX0, frameX1, pts1, pts2)
        cv2.circle (image, (x,y), 5, (0,255,0), 2)

def reset():
    global image, imageCopy
    image = imageCopy.copy()

def getCoord(mat):
	arr = np.array(mat)
	return [int(arr[0]/arr[2]), int(arr[1]/arr[2])]

def adjustColorLeft2Mid(src, dst):
    r1,r2,r3 = 1.0, 0.95, 0.95
    global h,w,c
    for i in range(h):
        for j in range(w):
            src[i][j] *= np.array([r1,r2,r3])

def adjustColorRight2Mid(src, dst):
    r1,r2,r3 = 0.90, 0.95, 0.94
    print r1,r2,r3
    global h,w,c
    for i in range(h):
        for j in range(w):
            src[i][j] *= np.array([r1,r2,r3])

np.set_printoptions(suppress=True)

# homography.left is calculated using frame number 5113
frameX0 = cv2.imread('frames/fr_l_5113.jpg')
frameX1 = cv2.imread('frames/fr_m_5113.jpg')

# homography.right is calculated using frame number 5
# frameX0 = cv2.imread('frame/fr_r_5.jpg')
# frameX1 = cv2.imread('frame/fr_m_5.jpg')

# save individual frames to disk
# saveFrames(vidName2, 0.5)
# saveFrames(vidName2, 0.5)
# saveFrames(vidName2, 0.5)

# points to store matched features. Note they must be of the same size and ordering
pts1,pts2   = [],[]
image       = drawMatchedKeypoints(frameX0, frameX1, pts1, pts2)
imageCopy   = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouseEvent)

while True:
    cv2.imshow('image',image)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # press "s" to show stiched image
    if key == ord("s"):
        reset()
        #make sure you have run "f"  getMatchedPointsFromImages
        #OR
        #you have manually selected matched features from images
        image       = stitchLeft2Mid(frameX0,frameX1,pts1,pts2)
        # image       = stitchRight2Mid(frameX0,frameX1,pts1,pts2)
    # press "f" to show automatically matched features
    if key == ord("f"):
        reset()
        pts1,pts2   = getMatchedPointsFromImages(frameX0, frameX1)
        image       = drawMatchedKeypoints(frameX0, frameX1, pts1, pts2)
    # press "c" to clear matches
    if key == ord("c"):
        clearPoints()
cv2.destroyAllWindows()