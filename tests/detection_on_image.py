import cv2
from batch_face import RetinaFace

detector = RetinaFace(gpu_id=0)
img = cv2.imread("examples/obama.jpg")
import time

start = time.time()
faces = detector(img, cv=True)
box, landmarks, score = faces[0]
print(time.time() - start)
