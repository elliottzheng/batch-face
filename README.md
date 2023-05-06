# Batch Face for Modern Research

## 🚧Documentation under construction, check tests folder for more details. 🚧

This repo provides the out-of-box face detection and face alignment with batch input support and enables real-time application on CPU.

## Features
1. Batch input support for faster data processing.
2. Smart API.
3. Ultrafast with inference runtime acceleration.
4. Automatically download pre-trained weights.
5. Minimal dependencies.

### Requirements

- Linux, Windows or macOS
- Python 3.5+ (it may work with other versions too)
- opencv-python
- PyTorch (>=1.0) 
- ONNX (optional)

While not required, for optimal performance it is highly recommended to run the code using a CUDA enabled GPU.

## Install

The easiest way to install it is using pip:

```bash
pip install git+https://github.com/elliottzheng/batch-face.git@master
```
No extra setup needs, most of the pretrained weights will be downloaded automatically.

If you have trouble install from source, you can try install from PyPI:

```bash
pip install batch-face
```
the PyPI version is not guaranteed to be the latest version, but we will try to keep it up to date.

## Usage
You can clone the repo and run tests like this
```
python -m tests.camera
```
### Face Detection

##### Detect face and five landmarks on single image
```python
import cv2
from batch_face import RetinaFace

detector = RetinaFace(gpu_id=0)
img = cv2.imread("examples/obama.jpg")
faces = detector(img, cv=True) # set cv to False for rgb input, the default value of cv is False
box, landmarks, score = faces[0]
```
##### Running on CPU/GPU

In order to specify the device (GPU or CPU) on which the code will run one can explicitly pass the device id.
```python
from batch_face import RetinaFace
# 0 means using GPU with id 0 for inference
# default -1: means using cpu for inference
detector = RetinaFace(gpu_id=0) 
```
|      | GPU(GTX 1080TI,batch size=1) | GPU(GTX 1080TI，batch size=750) | CPU(Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz) |
| ---- | ---------------------------- | ------------------------------- | --------------------------------------------- |
| FPS  | 44.02405810720893            | 96.64058005582535               | 15.452635835550483                            |
| SPF  | 0.022714852809906007         | 0.010347620010375976            | 0.0647138786315918                            |


##### Batch input for faster detection

**Detector with CUDA process batch input faster than the same amount of single input.** 

```python
import cv2
from batch_face import RetinaFace

detector = RetinaFace()
img= cv2.imread('examples/obama.jpg')[...,::-1]
all_faces = detector([img,img]) # return faces list of all images
box, landmarks, score = all_faces[0][0]
```

Note: All the input images must of the same size, for input images with different size, please use `detector.pseudo_batch_detect`.

![](./images/gpu_batch.png)

### Face Alignment
##### face alignment on single image

```python 
from batch_face import drawLandmark_multiple, LandmarkPredictor, RetinaFace

predictor = LandmarkPredictor(0)
detector = RetinaFace(0)

imgname = "examples/obama.jpg"
img = cv2.imread(imgname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = detector(img)

if len(faces) == 0:
    print("NO face is detected!")
    exit(-1)

# the first input for the predictor is a list of face boxes. [[x1,y1,x2,y2]]
results = predictor(faces, img, from_fd=True) # from_fd=True to convert results from our detection results to simple boxes

for face, landmarks in zip(faces, results):
    img = drawLandmark_multiple(img, face[0], landmarks)
```


## References

- Face Detection Network and pretrained model are from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- Face Alignment Network and pretrained model are from [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)
- Face Reconstruction Network and pretrained model are from [cleardusk/3DDFA](https://github.com/cleardusk/3DDFA)
