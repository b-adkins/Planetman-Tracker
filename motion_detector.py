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
ap.add_argument("-r", "--aspect_ratio", default='0.5, 0.9', help="min,max")
args = vars(ap.parse_args())
area_min, area_max = args['area'].split(':')
area_min = int(area_min) if area_min else 0
area_max = int(area_max) if area_max else 1e5
aspect_min, aspect_max = args['aspect_ratio'].split(',')
aspect_min = float(aspect_min)
aspect_max = float(aspect_max)
 
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
    
# initialize a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()
 
# Initialize camshift
def calcHistOfRect(frame, r, c, h, w):
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    # print mask
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

roi_hist = None

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
 
    
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
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40, 40))
    normed_illum = frame_lab.copy()
    normed_illum[:, :, 0] = cv2.equalizeHist(normed_illum[:, :, 0]) # Global
#    normed_illum[:, :, 0] = clahe.apply(normed_illum[:, :, 0]) # Local
    normed_illum = cv2.cvtColor(normed_illum, cv2.COLOR_LAB2BGR)
#    normed_illum = cv2.GaussianBlur(normed_illum, (9, 9), 0)
    
    # Color normalizaiton
    normed = frame_lab.copy()
    normed[:, :, 0] = 60* np.ones(normed[:, :, 0].shape, np.uint8)
    normed = cv2.cvtColor(normed, cv2.COLOR_LAB2BGR)
        
    # Adds the fgmask
    fgmask = fgbg.apply(normed_illum)
    thresh = fgmask
#    thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]  # Remove shadows
 
    thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, np.ones((3, 3),np.uint8))
#    thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_OPEN, np.ones((3, 3),np.uint8))
    
    
    # find contours processed black-and-white image
        # Canny edge detection may be useful later
    im_cnt, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
 
    # loop over the contours
    for c in cnts:   
        a = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)

        # Filter by size
        if a < area_min or a > area_max:
            continue
        
        # Filter by aspect ratio
        aspect_ratio = float(w)/h
        if aspect_ratio < aspect_min or aspect_ratio > aspect_max:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(frame, [c], 0, (0, 0, 255), 1)
        cv2.putText(frame, "{}={}x{}".format(a, w, h), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
        
        # Lazy initialize mean shift tracker on first contour
        if roi_hist is None:
            roi_hist = calcHistOfRect(normed_illum, y, x, h, w)
            track_window = (x, y, w, h)

    if roi_hist is not None:
        # Run meanshift
        hsv = cv2.cvtColor(normed_illum, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        if ret:
            x, y, w,h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2) 
        else:
            roi_hist = None
        
    # show the frame
    cv2.imshow("Video Feed", frame)
#    cv2.imshow("Illumination normalized", normed_illum)
    cv2.imshow("Color normalized", normed)
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
