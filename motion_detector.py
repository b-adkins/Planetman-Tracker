# import the necessary packages
import argparse
import datetime
import numpy as np
import sys
import time

import imutils
import cv2
 
cv2.ocl.setUseOpenCL(False)
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--area", default='500:', help="min:max, e.g. 200:500. Either can be blank")
args = vars(ap.parse_args())
area_min, area_max = args['area'].split(':')
area_min = int(area_min) if area_min else 0
area_max = int(area_max) if area_max else 1e5
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

if not camera.isOpened():
    print "Unable to open video source"
    sys.exit()
    
# initialize the first frame in the video stream
firstFrame = None

# initialize a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()
 

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, raw_frame) = camera.read()
 
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
 
    # Shrink the frame
    frame = imutils.resize(raw_frame, width=512)
    
    # Illumination normalization
    normed = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40, 40))
    normed[:, :, 0] = cv2.equalizeHist(normed[:, :, 0]) # Global
#    normed[:, :, 0] = clahe.apply(normed[:, :, 0]) # Local
#    normed[:, :, 0] = 60* np.ones(normed[:, :, 0].shape, np.uint8)
    normed = cv2.cvtColor(normed, cv2.COLOR_LAB2BGR)
#    normed = cv2.GaussianBlur(normed, (9, 9), 0)
 
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = normed
        continue
        
    # Adds the fgmask
    fgmask = fgbg.apply(normed)
    thresh = fgmask
#    thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]  # Remove shadows
 
    thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, np.ones((3, 3),np.uint8))
#    thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_OPEN, np.ones((3, 3),np.uint8))
    
    
    # find contours processed black-and-white image
    im_cnt, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
 
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        a = cv2.contourArea(c)
        if a < area_min or a > area_max:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(frame, [c], 0, (0, 0, 255), 1)
        cv2.putText(frame, "{}={}x{}".format(a, w, h), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
        
    # show the frame
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Illumination normalized", normed)
    cv2.imshow('Background subtraction', fgmask)
    cv2.imshow("Thresh", thresh)

    # if the `q` key is pressed, break from the lop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        while (cv2.waitKey(1) & 0xFF) != ord(" "):
            pass
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
