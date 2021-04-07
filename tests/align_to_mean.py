import cv2
from batch_face import RetinaFace
import numpy as np
import os
from skimage import transform

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    detector = RetinaFace(0)

    imgname = "examples/obama.png"
    img = cv2.imread(imgname)

    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    face = faces[0]

    box, landmark, score = face

    std_points_256 = np.array(
        [
            [85.82991, 85.7792],
            [169.0532, 84.3381],
            [127.574, 137.0006],
            [90.6964, 174.7014],
            [167.3069, 173.3733],
        ]
    )

    trans = transform.SimilarityTransform()

    res = trans.estimate(landmark, std_points_256)
    M = trans.params

    new_img = cv2.warpAffine(img, M[:2, :], dsize=(256, 256))

    cv2.imshow("new_img", new_img)
    cv2.waitKey(0)
