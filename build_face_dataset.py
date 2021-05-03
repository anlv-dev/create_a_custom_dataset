# import necessary libraries
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
from deepface import DeepFace

#construct the argument parser and parser the agurments

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--casade", required=True, help="path to where casade resides")
ap.add_argument("-o", "--output", required=True, help="path to output directory")

args =vars(ap.parse_args())

# load OpenCV's Haar casade for face detection from disk

#detector = cv2.CascadeClassifier(args["casade"])

detector = cv2.dnn.readNetFromCaffe(prototxt='models/deploy.prototxt.txt',
                                    caffeModel='models/res10_300x300_ssd_iter_140000.caffemodel')

# initial the video stream, allow the camera sensor to warm up
# and initila the total number of example faces written to disk
# thus far

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# loop over the frame from threaded video stream, clone it
# just in case we want to write it to disk the resize frame
# so we can apply face detection faster

while True:
    frame = vs.read()
    orig = frame.copy()
    size = (640,480)
    frame = imutils.resize(frame, size)

    #detect face in the grayscale frame

    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )

    # loop over the face detections and draw them on the frame
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'k' key was pressed, write the *orignial* frame to disk
    # so we can later process it, and use it for regconition
    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(
            str(total).zfill(5))])
        print(p)
        cv2.imwrite(p, orig)
        total +=1

    # if the q key pressed, break from the loop
    elif key == ord("q"):
        break

    # print the total faces saved and do a bit of cleanup

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()



