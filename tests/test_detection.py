import cv2
from batch_face.face_detection import RetinaFace

detector = RetinaFace(gpu_id=0)
img = cv2.imread("examples/obama.jpg")
import time

start = time.time()
for i in range(1000):
    faces = detector(img, cv=True)
    box, landmarks, score = faces[0]
print(time.time() - start)
