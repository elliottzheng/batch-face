import cv2
from contexttimer import Timer
from batch_face import drawLandmark_multiple, LandmarkPredictor, RetinaFace
import time


if __name__ == "__main__":
    file = r"E:\MaskedFace\finetune\PFLD_9.pth"
    predictor = LandmarkPredictor(0, backbone="PFLD", file=file)
    detector = RetinaFace(0)
    cap = cv2.VideoCapture(0)

    while True:
        start = time.time()
        ret, img = cap.read()
        faces = detector(img, cv=True, threshold=0.5)

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
