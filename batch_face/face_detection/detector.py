import os

import numpy as np
import torch

from .alignment import load_net, batch_detect


def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path


def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)


class RetinaFace:
    def __init__(
        self,
        gpu_id=-1,
        model_path=None,
        network="mobilenet",
    ):
        self.gpu_id = gpu_id
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device("cuda", gpu_id)
        )
        self.model = load_net(model_path, self.device, network)

    def detect(self, images, **kwargs):
        """
        cv_style: True if is bgr
        """
        kwargs["device"] = self.device
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return batch_detect(self.model, [images], **kwargs)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, **kwargs)
        elif isinstance(images, list):
            return batch_detect(self.model, np.array(images), **kwargs)
        elif isinstance(images, torch.Tensor):
            kwargs["is_tensor"] = True
            if len(images.shape) == 3:
                return batch_detect(self.model, images.unsqueeze(0), **kwargs)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, **kwargs)
        else:
            raise NotImplementedError()

    def pseudo_batch_detect(self, images, **kwargs):
        return [self.detect(image, **kwargs) for image in images]

    def __call__(self, images, **kwargs):
        return self.detect(images, **kwargs)
