import numpy as np
import cv2
from batch_face import RetinaFace, FarlParser, load_frames_rgb


if __name__ == "__main__":
    gpu_id = 0
    video_file = 'examples/ross.mp4'
    retinaface = RetinaFace(gpu_id)
    face_parser = FarlParser(gpu_id=gpu_id, name='farl/lapa/448') # optional "farl/celebm/448"
    label_names = face_parser.label_names
    print(label_names)
    
    frames = load_frames_rgb(video_file, max_frames=2)
    all_faces = retinaface(frames, return_dict=True, threshold=0.95)
    # optional, you can do some face filtering here
    all_faces = face_parser(frames, all_faces)
    for frame_i, (faces, frame) in enumerate(zip(all_faces, frames)):
        for face_i, face in enumerate(faces):
            seg_logits = face['seg_logits']
            seg_preds = face['seg_preds']
            vis_seg_preds = face_parser.color_lut[seg_preds]
            # blend with input frame
            frame = cv2.addWeighted(frame, 0.5, vis_seg_preds, 0.5, 0)
            vis_frame = np.concatenate([vis_seg_preds, frame], axis=1)
            cv2.imwrite(f'vis_{frame_i}_{face_i}.png', vis_frame[...,::-1])


