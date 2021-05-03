from deepface import DeepFace
import imutils
from imutils.video import VideoStream
import cv2
import time
import pandas as pd


backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
backends = backends[1]

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

while True:
    frame = vs.read()
    base_frame = frame.copy()
    frame = imutils.resize(frame, width=480)
    original_size = base_frame.shape
    detected_aligned_face = DeepFace.detectFace(img_path=frame, detector_backend=backends)
    print(len(detected_aligned_face))
    #print(detected_aligned_face)

    # show the output frame
    cv2.imshow("Frame", base_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'k' key was pressed, write the *orignial* frame to disk
    # so we can later process it, and use it for regconition
    if key == ord("k"):
        p = 'dataset/lvan/' + str(total) + '.png'
        print(p)
        cv2.imwrite(p, frame)
        total += 1

    # if the q key pressed, break from the loop
    elif key == ord("q"):
        break

    # print the total faces saved and do a bit of cleanup

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
