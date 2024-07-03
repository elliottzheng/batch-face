# Batch Face for Modern Research

## 🚧Documentation under construction, check tests folder for more details. 🚧

This repo provides the out-of-box face detection and face alignment with batch input support and enables real-time application on CPU.

## Features
1. Batch input support for faster data processing.
2. Smart API.
3. Ultrafast with inference runtime acceleration.
4. Automatically download pre-trained weights.
5. Minimal dependencies.
6. Unleash the power of GPU for batch processing.

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
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

max_size = 1080 # if the image's max size is larger than 1080, it will be resized to 1080, -1 means no resize
resize = 1 # resize the image to speed up detection, default is 1, no resize
threshold = 0.95 # confidence threshold

# now we recommand to specify return_dict=True to get the result in a more readable way
faces = detector(img, threshold=threshold, resize=resize, max_size=max_size, return_dict=True)
face = faces[0]
box = face['box']
kps = face['kps']
score = face['score']

# the old way to get the result
faces = detector(img, threshold=threshold, resize=resize, max_size=max_size)
box, kps, score = faces[0]

```
##### Running on CPU/GPU

In order to specify the device (GPU or CPU) on which the code will run one can explicitly pass the device id.
```python
from batch_face import RetinaFace
# 0 means using GPU with id 0 for inference
# default -1: means using cpu for inference
fp16 = True # use fp16 to speed up detection and save GPU memory

detector = RetinaFace(gpu_id=0, fp16=True)
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
img= cv2.imread('examples/obama.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

max_size = 1080 # if the image's max size is larger than 1080, it will be resized to 1080, -1 means no resize
resize = 1 # resize the image to speed up detection, default is 1, no resize
resize_device = 'cpu' # resize on cpu or gpu, default is gpu
threshold = 0.95 # confidence threshold for detection
batch_size = 100 # batch size for detection, the larger the faster but more memory consuming, default is -1, which means batch_size = number of input images
batch_images = [img,img] # pseudo batch input

all_faces = detector(batch_images, threshold=threshold, resize=resize, max_size=max_size, batch_size=batch_size) 
faces = all_faces[0] # the first input image's detection result
box, kps, score = faces[0] # the first face's detection result
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
### Head Pose Estimation
##### Head pose estimation on video
```python
from batch_face import RetinaFace, SixDRep, draw_landmarks, load_frames_rgb, Timer

vis = True
gpu_id = 0
batch_size = 100
threshold = 0.95
detector = RetinaFace(gpu_id=gpu_id)
head_pose_estimator = SixDRep(gpu_id=gpu_id)
video_file = 'examples/ross.mp4'
frames = load_frames_rgb(video_file) # simple wrapper to load video frames with opencv and convert to RGB, 0~255, UInt8, HWC
print(f'Loaded {len(frames)} frames')
print('image size:', frames[0].shape)
# it might take longer time to detect since is first time to run the model
all_faces = detector(frames, batch_size=batch_size, return_dict=True, threshold=threshold, resize=0.5)
head_poses = head_pose_estimator(all_faces, frames, batch_size=batch_size, update_dict=True, input_face_type='dict')
# the head pose will be updated in the all_faces dict
out_frames = []
for faces, frame in zip(all_faces, frames):
    for face in faces:
        head_pose_estimator.plot_pose_cube(frame, face['box'], **face['head_pose'])
    out_frames.append(frame)

if vis:
    import imageio
    out_file = 'examples/head_pose.mp4'
    imageio.mimsave(out_file, out_frames, fps=8)
```
check out the result video [here](./examples/head_pose.mp4)
you can run the script `python -m tests.video_head_pose` to see the result.


## References

- Face Detection Network and pretrained model are from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- Face Alignment Network and pretrained model are from [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)
- Face Reconstruction Network and pretrained model are from [cleardusk/3DDFA](https://github.com/cleardusk/3DDFA)
- Head Pose Estimation Network and pretrained model are from [jahongir7174/SixDRepNet](https://github.com/jahongir7174/SixDRepNet)
