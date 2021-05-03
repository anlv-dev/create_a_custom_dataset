import imutils
from imutils.video import VideoStream
import cv2
import time
import pandas as pd

detector = cv2.dnn.readNetFromCaffe(prototxt='models/deploy.prototxt.txt',
                                    caffeModel='models/res10_300x300_ssd_iter_140000.caffemodel')

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
    target_size = (300,300)

    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
    #print("aspect ratios x: ", aspect_ratio_x, ", y: ", aspect_ratio_y)
    target_frame = cv2.resize(frame, target_size)

    blog_image = cv2.dnn.blobFromImage(target_frame)
    # print(blog_image.shape)

    detector.setInput(blog_image)
    detections = detector.forward()
    detections = detections[0][0]
    #print(detections.shape)
    # print(detections.shape)

    df = pd.DataFrame(detections, columns=["image_id", "is_face", "confidence", "left", "top", "right", "bottom"])

    df = df[df["is_face"] == 1]
    df = df[df["confidence"] >= 0.9]
    print(df.head())
    for i, instance in df.iterrows():
        print(instance)
        confidence_score = str(round(100 * instance["confidence"], 2)) + " %"
        left = int(instance["left"] * 300)
        bottom = int(instance["bottom"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)

        detected_face = base_frame[int(top * aspect_ratio_y):int(bottom * aspect_ratio_y),
                        int(left * aspect_ratio_x):int(right * aspect_ratio_x)]

        print(detected_face.shape)

        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
            cv2.putText(base_frame, confidence_score, (int(left * aspect_ratio_x), int(top * aspect_ratio_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(base_frame, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)),
                          (int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (0, 255, 0), 1)

            print("Id ", i)
            print("Confidence: ", confidence_score)

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
