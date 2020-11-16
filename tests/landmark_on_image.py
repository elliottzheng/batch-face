import cv2
from batch_face.face_detection import RetinaFace
from batch_face import drawLandmark_multiple, LandmarkPredictor

if __name__ == "__main__":
    predictor = LandmarkPredictor(0)
    detector = RetinaFace(0)

    imgname = "examples/obama.jpg"
    img = cv2.imread(imgname)

    faces = detector(img, cv=True)

    if len(faces) == 0:
        print("NO face is detected!")
        exit(-1)

    results = predictor(faces, img, from_fd=True)

    for face, landmarks in zip(faces, results):
        img = drawLandmark_multiple(img, face[0], landmarks)

    cv2.imshow("", img)
    cv2.waitKey(0)
