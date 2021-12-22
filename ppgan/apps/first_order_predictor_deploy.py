#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import cycle
import os
import sys
import cv2
import tempfile
from numpy.lib.function_base import delete

import yaml
import imageio
import numpy as np
from tqdm import tqdm, trange
from pathlib import Path
from copy import deepcopy
from time import time
import numexpr as ne

import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.insert(0, os.path.dirname(root_path))

import paddle
from ppgan.utils.download import get_path_from_url
from ppgan.utils.animate import normalize_kp
from ppgan.modules.keypoint_detector import KPDetector
from ppgan.models.generators.occlusion_aware import OcclusionAwareGenerator
from ppgan.faceutils import face_detection
from ppgan.faceutils.face_detection.detection_utils import (
    upscale_detections,
    scale_bboxes,
)

# from gfpgan import GFPGANer
import moviepy.editor as mp

from ppgan.apps.base_predictor import BasePredictor
from PIL import Image
import imutils
import blend_modes as bm

from .ppdet.infer import *

# Global dictionary
SUPPORT_MODELS = {
    "YOLO",
    "RCNN",
    "SSD",
    "Face",
    "FCOS",
    "SOLOv2",
    "TTFNet",
    "S2ANet",
    "JDE",
    "FairMOT",
    "DeepSORT",
    "GFL",
    "PicoDet",
    "CenterNet",
}


class PredictConfig:
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, "infer_cfg.yml")
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf["arch"]
        self.preprocess_infos = yml_conf["Preprocess"]
        self.min_subgraph_size = yml_conf["min_subgraph_size"]
        self.labels = yml_conf["label_list"]
        self.mask = False
        self.use_dynamic_shape = yml_conf["use_dynamic_shape"]
        if "mask" in yml_conf:
            self.mask = yml_conf["mask"]
        self.tracker = None
        if "tracker" in yml_conf:
            self.tracker = yml_conf["tracker"]
        if "NMS" in yml_conf:
            self.nms = yml_conf["NMS"]
        if "fpn_stride" in yml_conf:
            self.fpn_stride = yml_conf["fpn_stride"]
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf["arch"]:
                return True
        raise ValueError(
            "Unsupported arch: {}, expect {}".format(yml_conf["arch"], SUPPORT_MODELS)
        )

    def print_config(self):
        print("-----------  Model Configuration -----------")
        print("%s: %s" % ("Model Arch", self.arch))
        print("%s: " % ("Transform Order"))
        for op_info in self.preprocess_infos:
            print("--%s: %s" % ("transform op", op_info["type"]))
        print("--------------------------------------------")


def load_detector(model_path):
    pred_config = PredictConfig(model_path)
    detector_func = "DetectorSOLOv2"

    detector = eval(detector_func)(
        pred_config,
        model_path,
        device="GPU",
        run_mode="fluid",
        batch_size=1,
    )
    return detector


class FirstOrderPredictor(BasePredictor):
    def __init__(
        self,
        output="output",
        weight_path=None,
        config=None,
        relative=False,
        adapt_scale=False,
        find_best_frame=False,
        best_frame=None,
        ratio=1.0,
        filename="result.mp4",
        face_detector="sfd",
        multi_person=False,
        image_size=256,
        face_enhancement=False,
        gfpgan_model_path=None,
        batch_size=1,
        mobile_net=False,
        preprocessing=True,
        face_align=False,
        solov_path=None,
    ):
        if config is not None and isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(config, dict):
            self.cfg = config
        elif config is None:
            if mobile_net:
                self.cfg = {
                    "model": {
                        "common_params": {
                            "num_kp": 10,
                            "num_channels": 3,
                            "estimate_jacobian": True,
                        },
                        "generator": {
                            "kp_detector_cfg": {
                                "temperature": 0.1,
                                "block_expansion": 32,
                                "max_features": 256,
                                "scale_factor": 0.25,
                                "num_blocks": 5,
                                "mobile_net": True,
                            },
                            "generator_cfg": {
                                "block_expansion": 32,
                                "max_features": 256,
                                "num_down_blocks": 2,
                                "num_bottleneck_blocks": 6,
                                "estimate_occlusion_map": True,
                                "dense_motion_params": {
                                    "block_expansion": 32,
                                    "max_features": 256,
                                    "num_blocks": 5,
                                    "scale_factor": 0.25,
                                },
                                "mobile_net": True,
                            },
                        },
                    }
                }
            else:
                self.cfg = {
                    "model": {
                        "common_params": {
                            "num_kp": 10,
                            "num_channels": 3,
                            "estimate_jacobian": True,
                        },
                        "generator": {
                            "kp_detector_cfg": {
                                "temperature": 0.1,
                                "block_expansion": 32,
                                "max_features": 1024,
                                "scale_factor": 0.25,
                                "num_blocks": 5,
                            },
                            "generator_cfg": {
                                "block_expansion": 64,
                                "max_features": 512,
                                "num_down_blocks": 2,
                                "num_bottleneck_blocks": 6,
                                "estimate_occlusion_map": True,
                                "dense_motion_params": {
                                    "block_expansion": 64,
                                    "max_features": 1024,
                                    "num_blocks": 5,
                                    "scale_factor": 0.25,
                                },
                            },
                        },
                    }
                }
        self.image_size = image_size
        if weight_path is None:
            if mobile_net:
                vox_cpk_weight_url = "https://paddlegan.bj.bcebos.com/applications/first_order_model/vox_mobile.pdparams"

            else:
                if self.image_size == 512:
                    vox_cpk_weight_url = "https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk-512.pdparams"
                else:
                    vox_cpk_weight_url = "https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk.pdparams"
            weight_path = get_path_from_url(vox_cpk_weight_url)

        self.weight_path = weight_path
        if not os.path.exists(output):
            os.makedirs(output)
        self.output = output
        self.filename = filename
        self.relative = relative
        self.adapt_scale = adapt_scale
        self.find_best_frame = find_best_frame
        self.best_frame = best_frame
        self.ratio = ratio
        self.face_detector = face_detector
        self.detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            face_detector=self.face_detector,
        )
        self.multi_person = multi_person
        self.face_enhancement = face_enhancement
        self.batch_size = batch_size

        self.generator, self.kp_detector = self.load_checkpoints(
            self.cfg, self.weight_path
        )
        self.solov2 = load_detector(solov_path)

        # from realesrgan import RealESRGANer
        # bg_upsampler = RealESRGANer(
        #         scale=2,
        #         model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        #         tile=400,
        #         tile_pad=10,
        #         pre_pad=0,
        #         half=True)
        # if gfpgan_model_path:
        #     self.gfpganer = GFPGANer(
        #         model_path=gfpgan_model_path,
        #         upscale=2,
        #         arch="clean",
        #         channel_multiplier=2,
        #         bg_upsampler=None,
        #     )
        # else:
        self.gfpganer = None

        if face_enhancement:
            from ppgan.faceutils.face_enhancement import FaceEnhancement

            self.faceenhancer = FaceEnhancement(batch_size=batch_size)
        self.detection_func = upscale_detections
        self.preprocessing = preprocessing
        self.face_alignment = face_align

    def read_img(self, path):
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # som images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]

        h, w, _ = img.shape
        if h >= 1024 or w >= 1024:
            if h > w:
                r = 1024.0 / h
                dim = (int(r * w), 1024)
            else:
                r = 1024.0 / w
                dim = (1024, int(r * h))
            img = cv2.resize(img, dim)
        return img

    def _add_border(self, image, border):
        image = Image.fromarray(image)
        border = Image.fromarray(border)
        image.paste(border, mask=border)
        return image

    def hover_frames_simplified(self, frames, hover):
        for i in frames.shape[0]:
            img_in_norm = frames[i] * (1 / 255.0)
            comp = 1.0 - (1.0 - img_in_norm[:, :, :3]) * (1.0 - hover[:, :, :3])
            frames[i][..., :3] = comp
            frames[i] *= 255.0
        return frames
    
    def hover_frames(self, frames, hover):
        return ne.evaluate("(1 - (1 - frames / 255) * (1 - hover / 255) ) * 255")

    def _decorate_frame(self, image, effect):
        return bm.screen(
            cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32), effect, 1.0
        ).astype(np.uint8)

    def _decorate(self, image, dim, effect=None, border=None):
        image = self._decorate_frame(image, effect)
        return self._add_border(image, dim, border)

    def _define_effects(self, frame_shape, effects, borders):
        h, w = frame_shape
        key = "landscape" if w > h + 40 else "portrait" if h > w + 40 else "square"
        if borders[key] is not None:
            border = cv2.cvtColor(cv2.imread(borders[key], -1), cv2.COLOR_BGR2RGBA)
        else:
            border = None
        if effects is not None:
            hover = cv2.cvtColor(cv2.imread(effects[key], -1), cv2.COLOR_BGR2RGBA)
        else:
            hover = None
        if borders[key] is not None:
            desired_height, desired_width = border.shape[:2]
            if key == "landscape":
                return (None, desired_height), border, hover
            elif key == "portrait":
                return (desired_width, None), border, hover
            else:
                return (desired_width, desired_height), border, hover
        else:
            return (w, h), None, None

    def decorate(self, frames, decoration):
        frame_shape = frames[0].shape[:2]
        borders = decoration["borders"]
        if ("hovers" in decoration.keys()) and (decoration["hovers"] is not None):
            effects = decoration["hovers"]
        else:
            effects = None
        s1 = time()
        dim, border, hover = self._define_effects(frame_shape, effects, borders)
        h, w = frame_shape

        orientation = (
            "landscape" if w > h + 40 else "portrait" if h > w + 40 else "square"
        )

        if hover is not None:
            s2 = time()
            print(w, h, "image size for hover")
            hover = cv2.resize(hover, (w, h)).astype(np.float32)

            # default hover
            # t = tqdm(frames, desc="Adding hovers to video", leave=True)
            # frames = [self._decorate_frame(frame, hover) for frame in t]

            # simplified hover
            #hover *= 1 / 255.0
            frames = np.array(frames).astype(np.float32)
            frames = self.hover_frames(frames, hover[..., :3])
            frames = frames.astype(np.uint8)
            s3 = time()
            print("hover time:", s3 - s2)
        if border is not None:
            s2 = time()
            if orientation == "landscape":
                border = imutils.resize(border, width=None, height=h)
                frames = self.fit_frames_to_landscape(np.array(frames), border)
            elif orientation == "portrait":
                border = imutils.resize(border, width=w, height=None)
                frames = self.fit_frames_to_portrait(np.array(frames), border)
            else:
                border = imutils.resize(border, width=w, height=h)
                frames = np.array(frames)

            if frames[0].shape[:2] != border.shape[:2]:
                border = cv2.resize(
                    border,
                    (frames[0].shape[1], frames[0].shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            s3 = time()
            print("border resize time:", s3 - s2)
            t = trange(frames.shape[0], desc="Adding borders to video", leave=True)
            frames = [self._add_border(frames[i], border) for i in t]
            s4 = time()
            print("add border time:", s4 - s3)
        return frames

    def write_with_audio(self, audio, out_frame, fps, decoration=None):
        s1 = time()
        out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_file.close()
        if decoration is not None:
            out_frame = self.decorate(out_frame, decoration)
        s2 = time()
        print("Decoration time: {}".format(s2 - s1))
        out_frame = [np.array(frame) for frame in out_frame]

        if audio is None:
            videoclip_1 = mp.ImageSequenceClip(out_frame, fps=fps)
            videoclip_1.write_videofile(out_file.name, preset="ultrafast", threads=4)
            s3 = time()
            print("No audio time: {}".format(s3 - s2))
        else:
            # audio_background = mp.AudioFileClip(audio)
            # # if audio.endswith(".mp3"):
            # #    audio_background = mp.AudioFileClip(audio)
            # # elif audio.endswith(".mp4"):
            # #    audio_background = mp.VideoFileClip(audio)
            # #    audio_background = audio_background.audio
            # temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            # temp.close()
            # imageio.mimsave(temp.name, [np.array(frame) for frame in out_frame], fps=fps)
            # videoclip_2 = mp.VideoFileClip(temp.name)
            # if audio_background.duration > videoclip_2.duration:
            #     audio_background = audio_background.subclip(0, videoclip_2.duration)
            # videoclip_2.set_audio(audio_background).write_videofile(out_file.name, audio_codec="copy")
            # os.remove(temp.name)
            videoclip_2 = mp.ImageSequenceClip(out_frame, fps=fps)
            videoclip_2.write_videofile(
                out_file.name, preset="ultrafast", audio=audio, audio_codec="aac", threads=4
            )

            print("Audio time: {}".format(time() - s2))
        return out_file.name

    def process_image(self, source_image, driving_videos, audio=None, decoration=None):

        driving_videos = deepcopy(driving_videos)

        img = np.array(source_image)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # som images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        h, w, _ = img.shape
        if h >= 768 or w >= 768:
            if h > w:
                r = 768.0 / h
                dim = (int(r * w), 768)
            else:
                r = 768.0 / w
                dim = (768, int(r * h))
            img = cv2.resize(img, dim)

        def get_prediction(face_image, driving_video):
            predictions = self.make_animation(
                face_image,
                driving_video,
                self.generator,
                self.kp_detector,
                relative=self.relative,
                adapt_movement_scale=self.adapt_scale,
            )
            return predictions

        st = time()
        results = []
        bboxes = self.extract_bbox(img.copy())
        s1 = time()
        print(s1 - st, "bbox step")
        # bboxes, coords = self.extract_bbox(img.copy())
        print(str(len(bboxes)) + " persons have been detected")
        areas = [x[4] for x in bboxes]
        indices = np.argsort(areas)
        bboxes = bboxes[indices]
        # coords = coords[indices]

        original_shape = img.shape[:2]
        if self.gfpganer:
            _, _, img = self.gfpganer.enhance(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        s2 = time()
        bboxes[:, :4] = scale_bboxes(
            original_shape, bboxes[:, :4].astype(np.float64), img.shape
        ).round()
        s3 = time()
        print(s3 - s2, "scale step")
        image_videos = []
        for driving_video in driving_videos[: len(bboxes)]:
            fps = driving_video["fps"]

            # try:
            #     driving_video["frames"] = [
            #         cv2.resize(im, (self.image_size, self.image_size)) / 255.0
            #         for im in driving_video["frames"]
            #     ]
            # except RuntimeError:
            #     print("Read driving video error!")
            #     pass

            image_videos.append(driving_video)
        s4 = time()
        print(s4 - s3, "resize driving video step")
        bbox2video = {}
        if len(bboxes) <= len(image_videos):
            bbox2video = {i: i for i in range(len(bboxes))}
        else:
            pool = cycle(range(len(image_videos)))
            bbox2video = {i: next(pool) for i in range(len(bboxes))}

        for i, rec in enumerate(bboxes):
            face_image = img.copy()[rec[1] : rec[3], rec[0] : rec[2]]
            face_image = (
                cv2.resize(face_image, (self.image_size, self.image_size)) / 255.0
            )
            predictions = get_prediction(
                face_image, image_videos[bbox2video[i]]["frames"]
            )
            results.append(
                {
                    "rec": rec,
                    "predict": [predictions[i] for i in range(predictions.shape[0])],
                }
            )
            if len(bboxes) == 1 or not self.multi_person:
                break
        s5 = time()
        print(s5 - s4, "get prediction step")
        out_frame = []

        box_masks = self.extract_masks(bboxes, img)
        s6 = time()
        print(s6 - s5, "extract masks step")
        patch = np.zeros(img.shape).astype("uint8")
        mask = np.zeros(img.shape[:2]).astype("uint8")

        for i in trange(max([len(i["frames"]) for i in image_videos])):
            frame = img.copy()

            for j, result in enumerate(results):
                x1, y1, x2, y2, _ = result["rec"]
                if i >= len(result["predict"]):
                    pass
                else:
                    out = result["predict"][i]
                    out = cv2.resize(out.astype(np.uint8), (x2 - x1, y2 - y1))

                    if len(results) == 1:
                        frame[y1:y2, x1:x2] = out
                        break
                    else:
                        patch[y1:y2, x1:x2] = out * np.dstack([(box_masks[j] > 0)] * 3)

                        mask[y1:y2, x1:x2] = box_masks[j]
                    frame = cv2.copyTo(patch, mask, frame)

            out_frame.append(frame)
            patch[:, :, :] = 0
            mask[:, :] = 0
        s7 = time()
        print(s7 - s6, "generate frame step")
        return self.write_with_audio(audio, out_frame, fps, decoration)

    def run(self, source_image, driving_videos_paths, filename, audio, decoration=None):

        self.filename = filename
        # videoclip_1 = mp.VideoFileClip(driving_video)
        # audio = videoclip_1.audio

        def get_prediction(face_image, driving_video):
            predictions = self.make_animation(
                face_image,
                driving_video,
                self.generator,
                self.kp_detector,
                relative=self.relative,
                adapt_movement_scale=self.adapt_scale,
            )
            return predictions

        source_image = self.read_img(source_image)

        results = []
        bboxes = self.extract_bbox(source_image.copy())
        # bboxes, coords = self.extract_bbox(source_image.copy())
        print(str(len(bboxes)) + " persons have been detected")
        areas = [x[4] for x in bboxes]
        indices = np.argsort(areas)
        bboxes = bboxes[indices]
        # coords = coords[indices]

        original_shape = source_image.shape[:2]
        if self.gfpganer:
            _, _, source_image = self.gfpganer.enhance(
                cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
            )
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        bboxes[:, :4] = scale_bboxes(
            original_shape, bboxes[:, :4].astype(np.float64), source_image.shape
        ).round()

        if isinstance(driving_videos_paths, str):
            if Path(driving_videos_paths).is_file():
                driving_videos_paths = [driving_videos_paths]
            elif Path(driving_videos_paths).is_dir():
                driving_videos_paths = [
                    str(filepath.absolute())
                    for filepath in Path(driving_videos_paths).glob("**/*.mp4")
                ]

        driving_videos = []
        for driving_video in driving_videos_paths[: len(bboxes)]:
            reader = imageio.get_reader(driving_video)
            fps = reader.get_meta_data()["fps"]

            try:
                driving_video = [
                    cv2.resize(im, (self.image_size, self.image_size)) / 255.0
                    for im in reader
                ]
            except RuntimeError:
                print("Read driving video error!")
                pass
            reader.close()

            driving_videos.append(driving_video)

        bbox2video = {}
        if len(bboxes) <= len(driving_videos):
            bbox2video = {i: i for i in range(len(bboxes))}
        else:
            pool = cycle(range(len(driving_videos)))
            bbox2video = {i: next(pool) for i in range(len(bboxes))}

        for i, rec in enumerate(bboxes):
            face_image = source_image.copy()[rec[1] : rec[3], rec[0] : rec[2]]
            face_image = (
                cv2.resize(face_image, (self.image_size, self.image_size)) / 255.0
            )
            predictions = get_prediction(face_image, driving_videos[bbox2video[i]])
            results.append(
                {
                    "rec": rec,
                    "predict": [predictions[i] for i in range(predictions.shape[0])],
                }
            )
            if len(bboxes) == 1 or not self.multi_person:
                break

        out_frame = []

        box_masks = self.extract_masks(bboxes, source_image)

        patch = np.zeros(source_image.shape).astype("uint8")
        mask = np.zeros(source_image.shape[:2]).astype("uint8")

        for i in trange(max([len(i) for i in driving_videos])):
            frame = source_image.copy()

            for j, result in enumerate(results):
                x1, y1, x2, y2, _ = result["rec"]

                if i >= len(result["predict"]):
                    pass
                else:
                    out = result["predict"][i]
                    out = cv2.resize(out.astype(np.uint8), (x2 - x1, y2 - y1))

                    if len(results) == 1:
                        frame[y1:y2, x1:x2] = out
                        break
                    else:
                        patch[y1:y2, x1:x2] = out * np.dstack([(box_masks[j] > 0)] * 3)

                        mask[y1:y2, x1:x2] = box_masks[j]
                    frame = cv2.copyTo(patch, mask, frame)

            out_frame.append(frame)
            patch[:, :, :] = 0
            mask[:, :] = 0

        self.write_with_audio(audio, out_frame, fps, decoration)

    def load_checkpoints(self, config, checkpoint_path):

        generator = OcclusionAwareGenerator(
            **config["model"]["generator"]["generator_cfg"],
            **config["model"]["common_params"],
            inference=True
        )

        kp_detector = KPDetector(
            **config["model"]["generator"]["kp_detector_cfg"],
            **config["model"]["common_params"]
        )

        checkpoint = paddle.load(self.weight_path)
        generator.set_state_dict(checkpoint["generator"])

        kp_detector.set_state_dict(checkpoint["kp_detector"])

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

    def make_animation(
        self,
        source_image,
        driving_video,
        generator,
        kp_detector,
        relative=True,
        adapt_movement_scale=True,
    ):
        with paddle.no_grad():

            predictions = []
            source = paddle.to_tensor(
                source_image[np.newaxis].astype(np.float32)
            ).transpose([0, 3, 1, 2])
            kp_source = kp_detector(source)
            kp_source_batch = {}
            kp_source_batch["value"] = paddle.tile(
                kp_source["value"], repeat_times=[self.batch_size, 1, 1]
            )
            kp_source_batch["jacobian"] = paddle.tile(
                kp_source["jacobian"], repeat_times=[self.batch_size, 1, 1, 1]
            )
            source = paddle.tile(source, repeat_times=[self.batch_size, 1, 1, 1])

            driving = paddle.to_tensor(
                np.array(driving_video[:1]).astype(np.float32)
            ).transpose([0, 3, 1, 2])
            kp_driving_initial = kp_detector(driving[:1])

            begin_idx = 0
            for _ in tqdm(
                range(int(np.ceil(float(len(driving_video)) / self.batch_size)))
            ):
                frame_num = min(self.batch_size, len(driving_video) - begin_idx)
                driving = paddle.to_tensor(
                    np.array(driving_video[begin_idx : begin_idx + frame_num]).astype(
                        np.float32
                    )
                ).transpose([0, 3, 1, 2])

                # driving_frame = driving[begin_idx: begin_idx + frame_num]
                kp_driving = kp_detector(driving)
                kp_source_img = {}
                kp_source_img["value"] = kp_source_batch["value"][0:frame_num]
                kp_source_img["jacobian"] = kp_source_batch["jacobian"][0:frame_num]

                kp_norm = normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale,
                )

                out = generator(
                    source[0:frame_num], kp_source=kp_source_img, kp_driving=kp_norm
                )
                img = np.transpose(out["prediction"].numpy(), [0, 2, 3, 1]) * 255.0
                if self.face_enhancement:
                    #     _, _, img = self.faceenhancer.enhance(img[0])
                    img = self.faceenhancer.enhance_from_batch(img)
                predictions.append(img)
                begin_idx += frame_num
        return np.concatenate(predictions)

    def extract_bbox(self, image):

        # frame = [image]
        predictions = self.detector.get_detections_for_image(np.array(image))
        predictions = list(
            filter(lambda x: ((x[3] - x[1]) * (x[2] - x[0])) > 1000, predictions)
        )
        # result, coords = self.detection_func(image, predictions)

        h, w, _ = image.shape
        predictions = self.detection_func(predictions, (0, 0, w, h))
        # predictions = list(map(lambda x: compute_aspect_preserved_bbox(x, image.shape[:2], 0.3), predictions))

        # return np.array(result), np.array(coords)
        return np.array(predictions)

    def extract_masks(self, bboxes, source_image):
        if len(bboxes) == 1:
            return
        box_masks = []
        for i, rec in enumerate(bboxes):
            face_image = source_image.copy()[rec[1] : rec[3], rec[0] : rec[2]]
            out = self.solov2.predict(image=[face_image.copy()])
            if out["segm"][0].shape != face_image.shape[:2]:
                out["segm"] = np.resize(
                    out["segm"], (out["segm"].shape[0], *face_image.shape[:2])
                )
            center = face_image.shape[0] // 2, face_image.shape[1] // 2
            mask = self.extract_mask(out, center)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            mask = cv2.dilate(mask, kernel)
            box_masks.append(mask)
        return box_masks

    def extract_mask(
        self,
        result,
        center,
        threshold=0.4,
    ):
        shape = result["segm"][0].shape

        idx = result["label"] == 0
        result["segm"] = result["segm"][idx]

        idx = result["score"][idx] >= 0.3
        result["segm"] = result["segm"][idx]

        if result["segm"].shape[0] > 0:
            # mask_idx = np.argmax(result["segm"].sum(axis=2).sum(axis=1))
            mask_idx = -1
            for i, mask in enumerate(result["segm"]):
                if mask[center[0], center[1]]:
                    mask_idx = i
            mask = result["segm"][mask_idx]
        else:
            mask = np.zeros(shape)

        return mask

    def fit_frames_to_landscape(self, frames, border):
        pad = (frames[0].shape[1] - border.shape[1]) // 2
        if pad > 0:
            frames = frames[:, :, pad:-pad]
        elif pad < 0:
            frames = list(frames)
            for i in trange(len(frames)):
                frames[i] = cv2.copyMakeBorder(
                    frames[i],
                    0,
                    0,
                    abs(pad),
                    abs(pad),
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
            frames = np.array(frames)
        return frames

    def fit_frames_to_portrait(self, frames, border):
        pad = (frames[0].shape[0] - border.shape[0]) // 2
        if pad > 0:
            frames = frames[:, pad:-pad, :]
        elif pad < 0:
            frames = list(frames)
            for i in trange(len(frames)):
                frames[i] = cv2.copyMakeBorder(
                    frames[i],
                    abs(pad),
                    abs(pad),
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
            frames = np.array(frames)
        return frames
