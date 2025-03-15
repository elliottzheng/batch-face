import torch
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import cv2

def convert2facer(all_faces_batchface, device='cpu'):
    if len(all_faces_batchface) == 0:
        return {
            "rects": torch.tensor([], device=device),
            "points": torch.tensor([], device=device),
            "scores": torch.tensor([], device=device),
            "image_ids": torch.tensor([], device=device)
        }
    
    if len(all_faces_batchface) > 0 and isinstance(all_faces_batchface[0], dict):
        all_faces_batchface = [all_faces_batchface]

    rects = []
    points = []
    scores = []
    image_ids = []
    for image_id, faces in enumerate(all_faces_batchface):
        for face in faces:
            rects.append(face["box"])
            points.append(face["kps"])
            scores.append(face["score"])
            image_ids.append(image_id)
    
    rects = np.stack(rects, axis=0)
    points = np.stack(points, axis=0)
    scores = np.stack(scores, axis=0)
    image_ids = np.stack(image_ids, axis=0)
    data = dict(rects=rects, points=points, scores=scores, image_ids=image_ids)
    data = {k: torch.from_numpy(v).to(device) for k, v in data.items()}
    return data




class FarlParser:
    def __init__(
        self,
        gpu_id: int = -1,
        name: Literal['farl/celebm/448', 'farl/lapa/448'] = 'farl/celebm/448',
        device: Literal['cuda', 'cpu', 'mps'] = 'cuda',
    ):
        
        self.gpu_id = gpu_id if device != "mps" else 0
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device(device, gpu_id)
        )
        try:
            import facer
            from facer.draw import _static_label_colors
            
        except:
            print("pyfacer is not installed, please install pyfacer first, checkout https://github.com/FacePerceiver/facer")
            raise
        from facer.version import __version__
        assert __version__ >= "0.0.5", "pyfacer version must be >= 0.0.5, checkout https://github.com/FacePerceiver/facer"
        
        self.face_parser = facer.face_parser(name=name, device=self.device)
        self.label_names = self.face_parser.label_names
        self.color_lut = (np.array(_static_label_colors) * 255).astype(np.uint8)

    def __call__(self, image, all_faces, return_keys = ['seg_preds', 'seg_logits']):
        '''
        return_keys: ['seg_preds', 'seg_logits']
        '''
        assert len(return_keys) >= 1, "return_keys must contain at least one key"

        return_logits = 'seg_logits' in return_keys
        return_preds = 'seg_preds' in return_keys

        all_faces_facer = convert2facer(all_faces, device=self.device)
        all_faces_facer = self.face_parser(image, all_faces_facer)
        counter = 0

        all_seg_logits = all_faces_facer['seg']['logits']
        all_seg_preds = all_seg_logits.argmax(dim=1).cpu().numpy()
        all_seg_logits = all_seg_logits.cpu().numpy()

        for faces in all_faces:
            for face in faces:
                face['seg_logits'] = all_seg_logits[counter] if return_logits else None
                face['seg_preds'] = all_seg_preds[counter] if return_preds else None
                counter += 1
        assert counter == len(all_seg_logits)
        return all_faces
    
    def vis_seg_preds(self, seg_preds, image):
        vis_seg_preds = self.color_lut[seg_preds]
        return cv2.addWeighted(image, 0.5, vis_seg_preds, 0.5, 0)
