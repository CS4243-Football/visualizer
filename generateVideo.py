import numpy as np
import cv2
import cv

fps = 7144.0/300
capSize = (4280,1017) # this is the size of my source video
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
vout = cv2.VideoWriter()
success = vout.open('output.mov',fourcc,fps,capSize,True) 

heightField = 540
widthField = 4280
heightTopdown = 477
widthTopdown = 711

for i in range(1500):
        field = cv2.imread("track_" + str(i) + ".jpg")
        topdown = cv2.imread("view_" + str(i) + ".jpg")
        
        combine = np.zeros((heightTopdown+heightField, widthField, 3),np.uint8)
        combine[:heightField, :widthField, :3] = field
        combine[heightField:heightField+heightTopdown, 1800:1800+widthTopdown, :3] = topdown

        print i
        vout.write(combine) 

# Release everything if job is finished
vout.release() 
vout = None