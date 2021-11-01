import os
import sys
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser

import imageio
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize

import face_alignment
from tqdm import tqdm

TEST_VIDEOS = []

REF_FPS = 50
REF_FRAME_SIZE = 360


def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)


def compute_increased_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    left = int(left - increase_area * width)
    top = int(top - increase_area * height)
    right = int(right + increase_area * width)
    bot = int(bot + increase_area * height)

    return (left, top, right, bot)


def crop_bbox_from_frames(frame_list, tube_bbox, min_frames=16, image_shape=(256, 256), min_size=200,
                          increase_area=0.1, aspect_preserving=True):
    frame_shape = frame_list[0].shape
    # Filter short sequences
    if len(frame_list) < min_frames:
        print("short sequences")
        return None, None
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top
    # Filter if it is too small
    if max(width, height) < min_size:
        print("too small")
        return None, None
    
    if aspect_preserving:
        left, top, right, bot = compute_aspect_preserved_bbox(tube_bbox, increase_area)
    else:
        left, top, right, bot = compute_increased_bbox(tube_bbox, increase_area)

    left = np.clip(left, 0, frame_shape[1])
    right = np.clip(right, 0, frame_shape[1])
    top = np.clip(top, 0, frame_shape[0])
    bot = np.clip(bot, 0, frame_shape[0])

    selected = [frame[top:bot, left:right] for frame in frame_list]
    if image_shape is not None:
        out = [img_as_ubyte(resize(frame, image_shape, anti_aliasing=True)) for frame in selected]
    else:
        out = selected
 
    return out, [left, top, right, bot]


def save(path, frames, format):
    if format == '.mp4':
        imageio.mimsave(path, frames)
    elif format == '.png':
        if os.path.exists(path):
            print ("Warning: skiping video %s" % os.path.basename(path))
            return
        else:
            os.makedirs(path)
        for j, frame in enumerate(frames):
            imageio.imsave(os.path.join(path, str(j).zfill(7) + '.png'), frames[j]) 
    else:
        print ("Unknown format %s" % format)
        exit()


def extract_bbox(frame, fa):
    bbox = fa.face_detector.detect_from_image(frame[..., ::-1])[0]
    return bbox


def store(frame_list, tube_bbox, video_id, args):
    out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=0,
                                            image_shape=args.image_shape, min_size=0,
                                            increase_area=args.increase)
    if out is None:
        return []

    name = video_id
    partition = 'test' if video_id in TEST_VIDEOS else 'train'
    save(os.path.join(args.out_folder, partition, name), out, args.format)
    # path = os.path.join(args.out_folder, partition, name)
    # imageio.mimsave(path, out, fps=REF_FPS)
    return [{'bbox': '-'.join(map(str, final_bbox)), 'start': 0, 'end': len(frame_list), 'fps': REF_FPS,
             'video_id': video_id, 'height': frame_list[0].shape[0], 'width': frame_list[0].shape[1], 'partition': partition}]


def process_video(video_id, args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    video_path = os.path.join(args.in_folder, video_id)
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    tube_bbox = None
    frame_list = []
    for i, frame in enumerate(reader):
        if i == 0:
            mult = frame.shape[0] / REF_FRAME_SIZE
            bbox = extract_bbox(
                resize(frame, (REF_FRAME_SIZE, int(frame.shape[1] / mult)), preserve_range=True), fa)
            bbox = bbox * mult
            left, top, right, bot, _ = bbox
            tube_bbox = bbox[:-1]
        frame_list.append(frame)
    return store(frame_list, tube_bbox, video_id, args)


def run(params):
    video_id, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False)
    return process_video(video_id, args, fa)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--in_folder", default='custom')
    parser.add_argument("--out_folder", default='custom_dataset')
    parser.add_argument("--increase", default=0.1, type=float,
                        help='Increase bbox by this amount')
    parser.add_argument("--format", default='.mp4',
                        help='Store format (.png, .mp4)')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")

    args = parser.parse_args()

    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(cur_path)[0]
    sys.path.append(root_path)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        os.makedirs(args.out_folder + '/train')
        os.makedirs(args.out_folder + '/test')

    print("Video preprocessing has started...\n")
    videos = sorted(os.listdir(args.in_folder))
    for v in tqdm(videos):
        try:
            process_video(v, args)
        except Exception as e:
            print(f"Error while processing video {v}: {e}")
