import cv2
import numpy as np
import torch


def drawLandmark_multiple(img, bbox, landmark):
    """
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    """
    img = cv2.UMat(img).get()
    x1, y1, x2, y2 = np.array(bbox)[:4].astype(np.int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x, y in np.array(landmark).astype(np.int):
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img


pretrained_urls = {
    "MobileNet": "https://github.com/elliottzheng/fast-alignment/releases/download/weights_v1/mobilenet_224_model_best_gdconv_external.pth",
    "PFLD": "https://github.com/elliottzheng/fast-alignment/releases/download/weights_v1/pfld_model_best.pth",
}


def load_weights(file, backbone):
    if file is None:
        assert backbone in pretrained_urls
        url = pretrained_urls[backbone]
        return torch.utils.model_zoo.load_url(url)
    else:
        return torch.load(file, map_location="cpu")


def is_image(x):
    if isinstance(x, np.ndarray) and len(x.shape) == 3 and x.shape[-1] == 3:
        return True
    else:
        return False


def is_box(x):
    try:
        x = np.array(x)
        assert len(x) == 4
        assert (x[2:] - x[:2]).min() > 0
        return True
    except:
        return False


def detection_adapter(all_faces, batch=False):
    if not batch:
        return [box for box, _, _ in all_faces]  # 是单层列表
    else:
        return [[box for box, _, _ in faces] for faces in all_faces]  # 双层列表
