import numpy as np
import cv2
import cv2.cv as cv

def getPlayerLocations(im,bg,mask=None):

    #im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.absdiff(im,bg)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if mask!=None:
        im = cv2.bitwise_and(im,mask)
        #cv2.drawContours(img,maskcontour,-1,(0,0,255),4)
    im = cv2.inRange(im, np.array(cv2.cv.Scalar(40)),np.array(cv2.cv.Scalar(255)))
    #thresh, im = cv2.threshold(im, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #im = cv2.morphologyEx(im, cv2.MORPH_OPEN, None, iterations = 2)
    #im = cv2.erode(im, None, iterations=3)
    k               = None#np.ones((3,3),np.uint8)
    im = cv2.dilate(im, k, iterations = 5)
    #img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    contour,hier = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if(len(contour)>0):
        bigBox = []
        groundPts = []
        MIN_AREA    = 250
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                box = cv2.boundingRect(cnt)
                groundPt = (box[0]+box[2]/2,box[1]+box[3])
                groundPts.append(groundPt)
                bigBox.append(box)
    return groundPts


def getColorLocations(im,bg,m=None):
    image       = im.copy()
    if m!=None:
        image   = cv2.bitwise_and(image,image,mask=m)
    hsv         = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    red         = cv2.inRange(hsv, np.array(cv2.cv.Scalar(0,100,100)),np.array(cv2.cv.Scalar(10,255,255)))
    blue        = cv2.inRange(hsv, np.array(cv2.cv.Scalar(60,100,0)),np.array(cv2.cv.Scalar(155,255,200)))
    yellow      = cv2.inRange(hsv, np.array(cv2.cv.Scalar(20,100,200)),np.array(cv2.cv.Scalar(50,255,255)))  
    red         = cv2.dilate(red, None, iterations = 10)
    blue        = cv2.dilate(blue, None, iterations = 10)
    yellow      = cv2.dilate(yellow, None, iterations = 10)
    
    redCnt,hier = cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    redPts          = []
    if(len(redCnt)>0):
        MIN_AREA        = 200
        yOffset         = 0
        for cnt in redCnt:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                box     = cv2.boundingRect(cnt)
                redPt   = (box[0]+box[2]/2,box[1]+box[3]+yOffset)
                redPts.append(redPt)

    blueCnt,hier = cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bluePts     = []
    if(len(blueCnt)>0):
        MIN_AREA    = 200
        yOffset     = 0
        for cnt in blueCnt:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                box     = cv2.boundingRect(cnt)
                bluePt = (box[0]+box[2]/2,box[1]+box[3]+yOffset)
                bluePts.append(bluePt)

    yellowCnt,hier = cv2.findContours(yellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    yellowPts   = []
    if(len(yellowCnt)>0):   
        MIN_AREA    = 140
        yOffset     = 10
        for cnt in yellowCnt:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                box     = cv2.boundingRect(cnt)
                yellowPt= (box[0]+box[2]/2,box[1]+box[3]+yOffset)
                yellowPts.append(yellowPt)            
    return redPts,bluePts,yellowPts


def mapPlayerLocations(locations,scale):
    global perspectiveM
    if locations==None or len(locations)==0:
        return []
    locationsF32    = []
    for pt in locations:
        pt      = [pt[0]/scale,pt[1]/scale]
        locationsF32.append(pt)
    locationsF32 = np.float32([locationsF32])
    mapPts      = cv2.perspectiveTransform(locationsF32, perspectiveM)
    return mapPts[0]
    

def getLabeledMap(fieldMap,redMapPts,blueMapPts,yellowMapPts):
    mapView     = fieldMap.copy()
    for pt in redMapPts:
        pt      = (int(pt[0]),int(pt[1]))
        cv2.circle(mapView,pt,2,(0,0,255),2)
    for pt in blueMapPts:
        pt      = (int(pt[0]),int(pt[1]))
        cv2.circle(mapView,pt,2,(255,0,0),2)
    for pt in yellowMapPts:
        pt      = (int(pt[0]),int(pt[1]))
        cv2.circle(mapView,pt,2,(0,255,255),2)
    return mapView
    

def play(step,scale,bg=None,mask=None):
    global vidName, fieldMap
    #invid = cv2.VideoCapture(vidName)
    #print invid.grab()
    basicImg = cv2.imread("output_frames/fr_0.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    height, width = np.shape(basicImg)
    height = int(height * scale)
    width = int(width * scale)
    #width  = int(invid.get(cv.CV_CAP_PROP_FRAME_WIDTH)*scale)
    #height = int(invid.get(cv.CV_CAP_PROP_FRAME_HEIGHT)*scale)
    #fps    = float(invid.get(cv.CV_CAP_PROP_FPS))
    vidLen = 5716

    #print width, height, vidLen

    if bg!=None:
        bg = cv2.resize(bg,(width,height))  
    if mask!=None:
        mask = cv2.resize(mask,(width,height))
        maskcontour,maskhier = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(vidLen-1): #last frame somehow corrupted
        img = cv2.imread("output_frames/fr_" + str(i) + ".jpg"); 
        if i%step == 0:
            img             = cv2.resize(img,(width,height))

            if bg != None :
                redPts,bluePts,yellowPts    = getColorLocations(img,bg,mask)
                redMapPts                   = mapPlayerLocations(redPts,scale)
                blueMapPts                  = mapPlayerLocations(bluePts,scale)
                yellowMapPts                = mapPlayerLocations(yellowPts,scale)

                for pt in redPts:
                    cv2.circle(img,pt,2,(0,0,255),2)
                for pt in bluePts:
                    cv2.circle(img,pt,2,(255,0,0),2)
                for pt in yellowPts:
                    cv2.circle(img,pt,2,(0,255,255),2)
                
                mapView                     = getLabeledMap(fieldMap,redMapPts,blueMapPts,yellowMapPts)
                cv2.imshow("map",mapView)
            cv2.putText(img,'frame: '+str(i),(10,10),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,0))
            cv2.imshow(vidName, img)
                        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    #del invid
    cv2.destroyAllWindows()

vidName                     = "football_mid.mov"
bgName                      = "background.png"
mapName                     = "topdownField.jpg"
bg                          = cv2.imread(bgName)
fieldMap                    = cv2.imread(mapName)

#mask of the field
mask                        = np.zeros((1080,1920),dtype='uint8')
mask[0:250][:]              = 255
mask[1010:][:]              = 255
mask                        = cv2.bitwise_not(mask)

# define 4 corresponding points in video and field map to calculate perspective transformation
# best select 4 corners of the field 
rectInVid                   = np.array([[898,122],[587,323],[2052,114],[2462,312]], dtype='float32')
rectInMap                   = np.array([[3,3],[112,371],[707,3],[598,371]],dtype='float32')
perspectiveM                = cv2.getPerspectiveTransform(rectInVid,rectInMap)

play(4,0.5,bg,mask)         #if bg and mask are None, play normal video


