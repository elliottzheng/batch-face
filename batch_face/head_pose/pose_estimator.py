from sixdrepnet import SixDRepNet
import sixdrepnet.utils as utils
from opencv_transforms import transforms as cv_transforms
import torch
import numpy as np

crop_resize = cv_transforms.Compose([cv_transforms.Resize(224),
                                    cv_transforms.CenterCrop(224)])

normalize = cv_transforms.Compose([cv_transforms.ToTensor(),
                                    cv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def chunk_generator(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def flatten(l):
    return [item for sublist in l for item in sublist]

def chunk_call(model, chunk_size, input_tensor):
    outputs = []
    for chunk in chunk_generator(input_tensor, chunk_size):
        outputs.append(model(chunk))
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    else:
        return flatten(outputs)

class SixDRep:
    def __init__(self, gpu_id: int= -1, dict_path: str='') -> None:
        self.model = SixDRepNet(gpu_id=gpu_id, dict_path=dict_path)
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(gpu_id))

    def plot_pose_cube(self, frame, box, yaw, pitch, roll):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)

        x_min = max(0, x_min-int(0.2*bbox_height))
        y_min = max(0, y_min-int(0.2*bbox_width))
        x_max = x_max+int(0.2*bbox_height)
        y_max = y_max+int(0.2*bbox_width)
        utils.plot_pose_cube(frame,  yaw, pitch, roll, x_min + int(.5*(x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

    def __call__(self, all_faces, frames, batch_size=None, input_face_type='tuple', update_dict=True):
        '''
        frames: list of np.ndarray, 0~255, uint8, rgb order
        batch_size: int, if None, no chunking
        input_face_type: str, 'tuple' or 'dict' or 'box'
        update_dict: bool, if True, update the input dictionary with head pose
        '''
        # if update_dict:
        #     assert input_face_type == 'dict', 'input_face_type should be dict when updating dictionary'

        assert len(frames) == len(all_faces)
        if batch_size is None:
            batch_size = len(all_faces) # no chunking
        imgs_for_model = []
        metas = []
        for faces, frame, i in zip(all_faces, frames, range(len(frames))):
            for j, face in enumerate(faces):
                if input_face_type == 'tuple':
                    box = face[0]
                elif input_face_type == 'dict':
                    box = face['box']
                elif input_face_type == 'box':
                    box = face
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                imgs_for_model.append(normalize(crop_resize(img)))
                metas.append((i, j, x_min, y_min, x_max, y_max, bbox_width, bbox_height))

                # pitch, yaw, roll = model.predict(img)
                # img = model.draw_axis(img, yaw, pitch, roll)
                # frame[y_min:y_max, x_min:x_max] = img

                # utils.plot_pose_cube(frame,  yaw, pitch, roll, x_min + int(.5*(
                #             x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

        imgs_for_model = torch.stack(imgs_for_model).to(self.device)
        with torch.no_grad():
            pred = chunk_call(self.model.model, batch_size, imgs_for_model)

        euler = utils.compute_euler_angles_from_rotation_matrices(pred)*180/np.pi
        p = euler[:, 0].cpu().detach().numpy()
        y = euler[:, 1].cpu().detach().numpy()
        r = euler[:, 2].cpu().detach().numpy()

        # reorganize the output
        outputs = [[] for _ in range(len(frames))]
        for (i, j, x_min, y_min, x_max, y_max, bbox_width, bbox_height), pitch, yaw, roll in zip(metas, p, y, r):
            #  utils.plot_pose_cube(frames[i], yaw, pitch, roll, x_min + int(.5*(x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
            head_pose = {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            }
            outputs[i].append(head_pose)
            if update_dict and input_face_type == 'dict':
                all_faces[i][j]['head_pose'] = head_pose
        for faces, output in zip(all_faces, outputs):
            assert len(faces) == len(output)
        return outputs