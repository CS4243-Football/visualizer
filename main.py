import cv2
import numpy as np
from cv2 import cv


def main():
    print "Main Function. Let the fun begin"

    video_name = 'centre_camera.mp4'
    background_name = 'background.jpg'
    #setup initial location of window

    # r, h, c, w = 200, 30, 300, 20
    # track_window = (c, r, w, h)
    #
    # background = get_background(video_name)
    #
    # # print "Background", background
    # cv2.imshow('background', background)
    # #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #
    # cv2.imwrite('background.jpg', background)

    foreground_extraction(video_name, background_name)


def foreground_extraction(video_name, background_name):
    cap = cv2.VideoCapture(video_name)
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
    background = cv2.imread(background_name)

    # for fr in range(1, frame_count):
    #
    #     print "Next Frame Number is ", cap.get(cv.CV_CAP_PROP_POS_FRAMES)
    #     ret, frame = cap.read()
    #
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    #     # print subtract_frame
    #
    #     # gray = cv2.cvtColor(subtract_frame, cv2.COLOR_BGR2GRAY)
    #
    #     cv2.imshow('gray', gray)
    #
    #     # cv2.waitKey(0)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break

    fgbg = cv2.BackgroundSubtractorMOG()
    while cap.isOpened():
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame, learningRate=1.0/10)

        cv2.imshow('Foreground', fgmask)
        cv2.imshow('Original', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def get_background(video_name):
    cap = cv2.VideoCapture(video_name)

    # frame_height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    # frame_width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print "There are ", frame_count, " frames"
    ret, first_frame = cap.read()
    frame_sum = np.float32(first_frame)

    for fr in range(1, frame_count):
        print "Next Frame Number is ", cap.get(cv.CV_CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        frame_sum += np.float32(frame)

    average_frame = frame_sum / frame_count

    return cv2.convertScaleAbs(average_frame)

main()