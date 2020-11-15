import cv2
from contexttimer import Timer
from batch_face import drawLandmark_multiple, RetinaFace, LandmarkPredictor
import time
import numpy as np

if __name__ == "__main__":
    predictor_weight = r"PFLD_for_mask.onnx"
    predictor = LandmarkPredictor(gpu_id="onnx", backbone="PFLD", file=predictor_weight)
    detector = RetinaFace(0)
    cap = cv2.VideoCapture(0)
    faces = None
    results = None
    while True:
        start = time.time()
        ret, img = cap.read()
        if not ret:
            break
        if faces is None:
            faces = detector(img, cv=True, threshold=0.5)
        else:
            ldm_new = results[0]
            (x1, y1), (x2, y2) = ldm_new.min(0), ldm_new.max(0)
            box_new = np.array([x1, y1, x2, y2])
            box_new[:2] -= 10
            box_new[2:] += 10
            faces = [[box_new, None, None]]

        if len(faces) == 0:
            print("NO face is detected!")
            continue
        else:
            with Timer() as timer:
                results = predictor(faces, img, from_fd=True)
            for face, landmarks in zip(faces, results):
                img = drawLandmark_multiple(img, face[0], landmarks)

        cv2.imshow("", img)
        cv2.waitKey(1)
        print("FPS=", 1 / (time.time() - start))
