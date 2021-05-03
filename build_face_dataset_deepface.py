import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd

detector = cv2.dnn.readNetFromCaffe(prototxt='models/deploy.prototxt.txt',
                                    caffeModel='models/res10_300x300_ssd_iter_140000.caffemodel')

img1 = cv2.imread('dataset/lvan/00010.png')
base_img = img1.copy()
original_size = img1.shape
target_size = (300, 300)
print("original image size: ", original_size)

aspect_ratio_x = (original_size[1] / target_size[1])
aspect_ratio_y = (original_size[0] / target_size[0])
print("aspect ratios x: ",aspect_ratio_x,", y: ", aspect_ratio_y)

img1 = cv2.resize(img1, (300, 300))
# print(img1)
plt.imshow(img1[:, :, ::-1])
plt.show()

blog_image = cv2.dnn.blobFromImage(img1)
#print(blog_image.shape)

detector.setInput(blog_image)
detections = detector.forward()
detections = detections[0][0]
detections.shape
#print(detections.shape)

df = pd.DataFrame(detections, columns=["image_id", "is_face", "confidence", "left", "top", "right", "bottom"])

df = df[df["is_face"] == 1]
df = df[df["confidence"] >=0.9]
for i, instance in df.iterrows():
    print(instance)
    confidence_score = str(round(100 * instance["confidence"], 2)) + " %"
    left = int(instance["left"] * 300)
    bottom = int(instance["bottom"] * 300)
    right = int(instance["right"] * 300)
    top = int(instance["top"] * 300)

    detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
        cv2.putText(base_img, confidence_score, (int(left * aspect_ratio_x), int(top * aspect_ratio_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(base_img, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)),
                      (int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (255, 255, 255), 1)

        print("Id ", i)
        print("Confidence: ", confidence_score)
        # detected_face = cv2.resize(detected_face, (224, 224))
        plt.imshow(base_img[:, :, ::-1])
        plt.axis('off')
        plt.show()

