import cv2
from batch_face import (
    RetinaFace,
    drawLandmark_multiple,
    LandmarkPredictor,
)
import os

if __name__ == "__main__":
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    predictor = LandmarkPredictor(0, "PFLD")
    detector = RetinaFace(0)
    all_images = []
    names = os.listdir("examples")
    for name in names:
        img = cv2.imread(os.path.join("examples", name))
        all_images.append(img)

    all_faces = detector.pseudo_batch_detect(
        all_images, cv=True, threshold=0.9
    )  # 图不一样大 就只有伪 batch input

    all_results = predictor(all_faces, all_images, from_fd=True)

    assert len(all_results) == len(all_faces)

    for faces, landmarks, img, name in zip(all_faces, all_results, all_images, names):
        assert len(faces) == len(landmarks)
        for face, landmark in zip(faces, landmarks):
            img = drawLandmark_multiple(img, face[0], landmark)
        cv2.imwrite(os.path.join(output_dir, name), img)
