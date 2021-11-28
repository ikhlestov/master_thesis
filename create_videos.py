import os

import imageio
from tqdm import tqdm


base_dir = '/Users/IllarionK/Projects/master_degree'
if not os.path.exists(base_dir):
    base_dir = '/src'
frames_dir = os.path.join(base_dir, 'frames')
os.makedirs(frames_dir, exist_ok=True)
video_dir = os.path.join(base_dir, 'videos')
os.makedirs(video_dir, exist_ok=True)


for re_dir in tqdm(os.listdir(frames_dir)):
    try:
        video_path = os.path.join(video_dir, re_dir + '.mp4')
        if os.path.exists(video_path):
            continue
        re_frames_dir = os.path.join(frames_dir, re_dir)
        if not os.path.isdir(re_frames_dir):
            continue
        with imageio.get_writer(video_path, mode='I') as writer:
            for image_path in sorted(os.listdir(re_frames_dir), key=lambda x: int(x.split('.')[0])):
                image = imageio.imread(os.path.join(re_frames_dir, image_path))
                writer.append_data(image)
    except Exception as e:
        print(f"Skipped, {e}")
