from batch_face.utils import bbox_from_pts
import cv2
from batch_face import drawLandmark_multiple, RetinaFace, LandmarkPredictor
from batch_face.face_reconstruction import ShapeRegressor, parse_param
import time
import numpy as np

import time


def faces_from_results(results):
    faces = []
    for res in results:
        ldm_new = res
        box_new = bbox_from_pts(ldm_new)
        faces.append([box_new, None, None])
    return faces


if __name__ == "__main__":
    predictor = LandmarkPredictor(backbone="PFLD", file=None)
    detector = RetinaFace(0)
    regressor = ShapeRegressor(gpu_id=0)
    cap = cv2.VideoCapture(0)
    faces = None
    results = None
    recon_landmarks = []
    while True:
        ret, img = cap.read()
        start = time.time()
        if not ret:
            break
        if faces is None:
            faces = detector(img, cv=True, threshold=0.5)

        img_for_show = img.copy()
        if len(faces) == 0:
            print("NO face is detected!")
            continue
        else:
            if results is None:
                results = predictor(faces, img, from_fd=True)
            faces = faces_from_results(results)
            boxes = [face[0] for face in faces]
            recon_results = regressor(boxes, img)
            results = []
            for recon_res, box in zip(recon_results, boxes):
                pts = recon_res["pts68"]
                results.append(pts)
                img_for_show = drawLandmark_multiple(img_for_show, box, pts)

        cv2.imshow("", img_for_show)
        print("FPS=", 1 / (time.time() - start))
        cv2.waitKey(1)
