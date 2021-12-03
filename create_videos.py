import os

import click
import imageio
from tqdm import tqdm


BASE_DIR = '/Users/IllarionK/Projects/master_degree'
if not os.path.exists(BASE_DIR):
    BASE_DIR = '/src'


def process(frames_dir, rewrite=False):
    frames_dir_path = os.path.join(BASE_DIR, frames_dir)
    video_dir_path = os.path.join(BASE_DIR, frames_dir.replace('frames', 'videos'))
    os.makedirs(video_dir_path, exist_ok=True)

    for exp_name in tqdm(os.listdir(frames_dir_path)):
        try:
            video_file_path = os.path.join(video_dir_path, exp_name + '.mp4')
            if os.path.exists(video_file_path) and not rewrite:
                continue
            exp_frames_dir = os.path.join(frames_dir, exp_name)
            if not os.path.isdir(exp_frames_dir):
                continue
            with imageio.get_writer(video_file_path, mode='I') as writer:
                for image_path in sorted(os.listdir(exp_frames_dir), key=lambda x: int(x.split('.')[0])):
                    image = imageio.imread(os.path.join(exp_frames_dir, image_path))
                    writer.append_data(image)
        except Exception as e:
            print(f"Skipped, {e}")



@click.command()
@click.option('--frames_dir', '-f', required=True, type=click.Path(exists=True))
@click.option('--rewrite', '-r', is_flag=True)
def main(frames_dir, rewrite):
    process(frames_dir, rewrite)


if __name__ == '__main__':
    main()
