import cv2
from batch_face import (
    RetinaFace,
    drawLandmark_multiple,
)
import time
import os

if __name__ == "__main__":
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    detector = RetinaFace(0)
    all_images = []
    names = os.listdir("examples")
    img_names = []
    for name in names:
        if 'obama' in name or 'trump' in name:
            img = cv2.imread(os.path.join("examples", name))
            all_images.append(img)
            img_names.append(name)

    all_images = all_images * 20
    
    start = time.time()
    for i in range(10):
        all_faces = detector(
            all_images, cv=True, threshold=0.9, max_size = 640
        )  # 图不一样大 就只有伪 batch input
    print('new implementation time:', time.time() - start)

    for faces, img, img_name in zip(all_faces, all_images, img_names):
        for face in faces:
            img = drawLandmark_multiple(img, face[0])
        cv2.imwrite(os.path.join(output_dir, img_name), img)


