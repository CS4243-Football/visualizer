import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2.cv as cv

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
    r1,r2,r3 = 1.1, 0.95, 0.95 # these numbers are obtained empirically
    global h,w,c
    for i in range(h):
        for j in range(w):
            src[i][j] *= np.array([r1,r2,r3])

def adjustColorRight2Mid(src, dst):
    r1,r2,r3 = 0.79, 0.89, 0.9 # these numbers are obtained empirically
    global h,w,c
    for i in range(h):
        for j in range(w):
            src[i][j] *= np.array([r1,r2,r3])

H_left, H_right = readHomo()
print H_left
print H_right

# Define the codec and create VideoWriter object, this is only used when desired output is video
# fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
# out = cv2.VideoWriter('output.mov',fourcc, 24, (2880,540))

for i in range(7144): # 7144 is the total frame count
    frame_0 = cv2.imread('frames/fr_l_{}.jpg'.format(i))
    frame_1 = cv2.imread('frames/fr_m_{}.jpg'.format(i))
    frame_2 = cv2.imread('frames/fr_r_{}.jpg'.format(i))

    h,w,c = frame_0.shape
    adjustColorLeft2Mid(frame_0, frame_1)
    adjustColorRight2Mid(frame_2, frame_1)

    frame_0_warped = cv2.warpPerspective(frame_0,H_left,(w,h))
    frame_2_warped = cv2.warpPerspective(frame_2,H_right,(w * 2,h))

    final = np.zeros((h,w*3,c),np.uint8)
    final[:, :w, :] = frame_0_warped
    final[:, w:w*2, :] = frame_1
    final[:, w*2:, :] = frame_2_warped[:, w:, :]

    cv2.imshow('final', final)
    cv2.waitKey(30)
    cv2.imwrite('output_frames/fr_{}.jpg'.format(i), final)

cv2.destroyAllWindows()