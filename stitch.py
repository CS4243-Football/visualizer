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

def getHomo(l):
    l = l.split(" ")
    H = np.array([[l[0], l[1], l[2]], [l[3], l[4], l[5]], [l[6], l[7], l[8]]])
    return H.astype(float)

def readHomo():
    f = open('homography.left', 'r')
    line = f.readline()
    H_left = getHomo(line)
    f.close()
    f = open('homography.right', 'r')
    line = f.readline()
    H_right = getHomo(line)
    f.close()
    return H_left, H_right

def adjustColorLeft2Mid(src, dst):
    src_wd = src[250:270,800:820,:]
    dst_wd = dst[210:230,60:80,:]
    r1,r2,r3 = getRatio(src_wd,dst_wd)
    print r1,r2,r3
    global h,w,c
    for i in range(h):
        for j in range(w):
            src[i][j] *= np.array([r1,r2,r3])

def adjustColorRight2Mid(src, dst):
    src_wd = src[295:395,105:145,:]
    dst_wd = dst[310:410,840:880,:]
    r1,r2,r3 = getRatio(src_wd,dst_wd)
    print r1,r2,r3
    global h,w,c
    for i in range(h):
        for j in range(w):
            src[i][j] *= np.array([r1,r2,r3])

def getRatio(src, dst):
    srcAvg = getAvg(src)
    dstAvg = getAvg(dst)
    r1 = dstAvg[0]/srcAvg[0]
    r2 = dstAvg[1]/srcAvg[1]
    r3 = dstAvg[2]/srcAvg[2]
    return r1,r2,r3

def getAvg(img):
    s1 = 0.
    s2 = 0.
    s3 = 0.
    for row in img:
        for pt in row:
            s1 += pt[0]
            s2 += pt[1]
            s3 += pt[2]
    n = img.shape[0]*img.shape[1]
    return [s1/n,s2/n,s3/n]

H_left, H_right = readHomo()
print H_left
print H_right

frame_0 = cv2.imread('fr_l_5.jpg')
frame_1 = cv2.imread('fr_m_5.jpg')
frame_2 = cv2.imread('fr_r_5.jpg')

h,w,c = frame_0.shape
adjustColorLeft2Mid(frame_0, frame_1)
adjustColorRight2Mid(frame_2, frame_1)

frame_0_warped = cv2.warpPerspective(frame_0,H_left,(w,h))
frame_2_warped = cv2.warpPerspective(frame_2,H_right,(w*2,h))

final = np.zeros((h,w*3,c),np.uint8)
final[:,:w,:] = frame_0_warped
final[:,w:w*2,:] = frame_1
final[:,w*2:,:] = frame_2_warped[:,w:,:]

cv2.imwrite('final.jpg', final)