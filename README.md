# Batch Face for Modern Research

## 🚧Under Construction, not available now. 🚧

This repo provides the out-of-box face detection and face alignment with batch input support and enables real-time application on CPU.

## Features
1. Batch input support for faster data processing.
2. Smart API.
3. Ultra fast.

### Requirements

- Python 3.5+ (it may work with other versions too)
- Linux, Windows or macOS
- PyTorch (>=1.0)

While not required, for optimal performance it is highly recommended to run the code using a CUDA enabled GPU.

## Install

The easiest way to install it is using pip:

```bash
pip install batch-face
```
or
```bash
pip install git+https://github.com/elliottzheng/batch-face.git@master
```

## Usage
##### Detect face and five landmarks on single image
```python
import cv2
from batch_face import RetinaFace

detector = RetinaFace()
img= cv2.imread('examples/obama.jpg')[...,::-1]
faces = detector(img)
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

All the input images must of the same size.

**Detector with CUDA process batch input faster than the same amount of single input.** 

```python
import cv2
from batch_face import RetinaFace

detector = RetinaFace()
img= cv2.imread('examples/obama.jpg')[...,::-1]
all_faces = detector([img,img]) # return faces list of all images
box, landmarks, score = all_faces[0][0]
```

![](./images/gpu_batch.png)

## Reference

- Network and pretrained model are from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
