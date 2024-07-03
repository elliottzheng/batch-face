from batch_face import RetinaFace, SixDRep, draw_landmarks, load_frames_rgb, Timer


if __name__ == '__main__':
    gpu_id = 0
    batch_size = 100
    threshold = 0.95
    detector = RetinaFace(gpu_id=gpu_id)
    head_pose_estimator = SixDRep(gpu_id=gpu_id)
    video_file = 'examples/ross.mp4'
    with Timer('load_frames'):
        frames = load_frames_rgb(video_file)
    print(f'Loaded {len(frames)} frames')
    print('image size:', frames[0].shape)
    # it might take longer time to detect since is first time to run the model
    with Timer('detector'):
        all_faces = detector(frames, batch_size=batch_size, return_dict=True, threshold=threshold, resize=0.5)
    with Timer('head_pose_estimator'):
        head_poses = head_pose_estimator(all_faces, frames, batch_size=batch_size, update_dict=True, input_face_type='dict')
    out_frames = []
    for faces, frame in zip(all_faces, frames):
        for face in faces:
            head_pose_estimator.plot_pose_cube(frame, face['box'], **face['head_pose'])
            frame = draw_landmarks(frame,landmark=face['kps'])
        out_frames.append(frame)
    vis = True
    if vis:
        import imageio
        out_file = 'examples/head_pose.mp4'
        imageio.mimsave(out_file, out_frames, fps=8)


