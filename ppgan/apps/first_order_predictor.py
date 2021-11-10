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

import os
import sys
import cv2

import yaml
import imageio
import time
import numpy as np
from tqdm import tqdm, trange

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
from ppgan.faceutils.mask.face_parser import FaceParser
from ppgan.faceutils.face_segmentation.face_seg import FaceSeg
from ppgan.faceutils.face_detection.detection_utils import union_results, polygon2mask, polygon2ellipsemask
from gfpgan import GFPGANer
import moviepy.editor as mp

from ppgan.apps.base_predictor import BasePredictor


class FirstOrderPredictor(BasePredictor):
    def __init__(self,
                 output='output',
                 weight_path=None,
                 config=None,
                 relative=False,
                 adapt_scale=False,
                 find_best_frame=False,
                 best_frame=None,
                 ratio=1.0,
                 filename='result.mp4',
                 face_detector='sfd',
                 multi_person=False,
                 image_size=256,
                 face_enhancement=False,
                 gfpgan_model_path=None, 
                 batch_size=1,
                 mobile_net=False, 
                 preprocessing=True,
                 face_align=False):
        if config is not None and isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(config, dict):
            self.cfg = config
        elif config is None:
            self.cfg = {
                'model': {
                    'common_params': {
                        'num_kp': 10,
                        'num_channels': 3,
                        'estimate_jacobian': True
                    },
                    'generator': {
                        'kp_detector_cfg': {
                            'temperature': 0.1,
                            'block_expansion': 32,
                            'max_features': 1024,
                            'scale_factor': 0.25,
                            'num_blocks': 5
                        },
                        'generator_cfg': {
                            'block_expansion': 64,
                            'max_features': 512,
                            'num_down_blocks': 2,
                            'num_bottleneck_blocks': 6,
                            'estimate_occlusion_map': True,
                            'dense_motion_params': {
                                'block_expansion': 64,
                                'max_features': 1024,
                                'num_blocks': 5,
                                'scale_factor': 0.25
                            }
                        }
                    }
                }
            }
        self.image_size = image_size
        if weight_path is None:
            if mobile_net:
                vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox_mobile.pdparams'

            else:
                if self.image_size == 512:
                    vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk-512.pdparams'
                else:
                    vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk.pdparams'
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
        self.multi_person = multi_person
        self.face_enhancement = face_enhancement
        self.batch_size = batch_size
        start = time.time()
        self.generator, self.kp_detector = self.load_checkpoints(
            self.cfg, self.weight_path)
      
        
        # from realesrgan import RealESRGANer
        # bg_upsampler = RealESRGANer(
        #         scale=2,
        #         model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        #         tile=400,
        #         tile_pad=10,
        #         pre_pad=0,
        #         half=True)
        if gfpgan_model_path: 
            self.gfpganer = GFPGANer(model_path=gfpgan_model_path, 
                                            upscale = 2, 
                                            arch = 'clean',
                                            channel_multiplier = 2,
                                            bg_upsampler = None)
        else:
            self.gfpganer = None
        print("model loading" , time.time() - start)
        if face_enhancement:
            from ppgan.faceutils.face_enhancement import FaceEnhancement
            self.faceenhancer = FaceEnhancement(batch_size=batch_size)
            # self.faceenhancer =  GFPGANer(model_path=gfpgan_model_path, 
            #                              upscale = 2, 
            #                              arch = 'clean',
            #                              channel_multiplier = 2,
            #                              bg_upsampler = None)
        self.detection_func = union_results
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
                dim = (1024, int(r*h))
            img = cv2.resize(img, dim)
        return img

    def write_with_audio(self, audio, out_frame, fps):
        if audio is None:
            imageio.mimsave(os.path.join(self.output, self.filename),
                            [frame for frame in out_frame],
                            fps=fps)
        else:
            temp = 'tmp.mp4'
            imageio.mimsave(temp,
                            [frame for frame in out_frame],
                            fps=fps)
            videoclip_2 = mp.VideoFileClip(temp)
            videoclip_2.set_audio(audio).write_videofile(os.path.join(self.output, self.filename),
                                                            audio_codec="aac")
            os.remove(temp)


    def run(self, source_image, driving_video, filename):
        
        self.filename = filename
        videoclip_1 = mp.VideoFileClip(driving_video)
        audio = videoclip_1.audio
        def get_prediction(face_image):
            predictions = self.make_animation(
                    face_image,
                    driving_video,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale)
            return predictions

        source_image = self.read_img(source_image)
        # if self.gfpganer:
        #     _, _, source_image = self.gfpganer.enhance(cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
        #     source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()['fps']

        
        try:
            driving_video = [cv2.resize(im, (self.image_size, self.image_size)) / 255.0  for im in reader]
        except RuntimeError:
            print("Read driving video error!")
            pass
        reader.close()

        # driving_video = [
        #     cv2.resize(frame, (self.image_size, self.image_size)) / 255.0 for frame in raw_driving_video
        # ]
        results = []
        start = time.time()
        bboxes, coords = self.extract_bbox(source_image.copy())
        print("extract bboxes", time.time() - start)
        print(str(len(bboxes)) + " persons have been detected")
        areas = [x[4] for x in bboxes]
        indices = np.argsort(areas)
        bboxes = bboxes[indices]
        coords = coords[indices]
        
        original_shape = source_image.shape[:2]
        if self.gfpganer:
            _, _, source_image = self.gfpganer.enhance(cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        bboxes[:, :4] = scale_bboxes(original_shape, bboxes[:, :4].astype(np.float64), source_image.shape).round()
        for i, c in enumerate(coords):
            coords[i] = scale_coords(original_shape, np.array(c).astype(np.float64), source_image.shape).round()
            coords[i] = list(coords[i])

        face_image = source_image.copy()
        for rec in bboxes:
            face_image = source_image.copy()[rec[1]:rec[3], rec[0]:rec[2]]
            face_image = cv2.resize(face_image, (self.image_size, self.image_size)) / 255.0
            predictions = get_prediction(face_image)
            results.append({'rec': rec, 'predict': [predictions[i] for i in range(predictions.shape[0])]})
            if len(bboxes) == 1 or not self.multi_person:
                break
        out_frame = []
        start = time.time()
        box_masks = self.extract_masks(results, coords, source_image)
        print("masks extraction: ", time.time()-start)
        start = time.time()

        patch = np.zeros(source_image.shape).astype('uint8')
        mask = np.zeros(source_image.shape[:2]).astype('uint8')
        for i in trange(len(driving_video)):
            frame = source_image.copy()
            # patch = np.zeros(frame.shape).astype('uint8')
            # mask = np.zeros(frame.shape[:2]).astype('uint8')
            for j, result  in enumerate(results):
                x1, y1, x2, y2, _ = result['rec']

                out = result['predict'][i]
                out = cv2.resize(out.astype(np.uint8), (x2-x1, y2-y1))
        
                if len(results) == 1:
                    frame[y1:y2, x1:x2] = out
                    break
                else: 
                    #patch = np.zeros(frame.shape).astype('uint8')
                    patch[y1:y2, x1:x2] = out * np.dstack([(box_masks[j] > 0)]*3)
                    
                    #mask = np.zeros(frame.shape[:2]).astype('uint8')
                    mask[y1:y2, x1:x2] = box_masks[j]
                frame = cv2.copyTo(patch, mask, frame)
             
            out_frame.append(frame)
            patch[:, :, :] = 0
            mask[:, :] = 0          

        print("video stitching", time.time() - start)
        start = time.time()
        self.write_with_audio(audio, out_frame, fps)
        print("video writing", time.time() - start)


    def load_checkpoints(self, config, checkpoint_path):

        generator = OcclusionAwareGenerator(
            **config['model']['generator']['generator_cfg'],
            **config['model']['common_params'], inference=True)

        kp_detector = KPDetector(
            **config['model']['generator']['kp_detector_cfg'],
            **config['model']['common_params'])

        checkpoint = paddle.load(self.weight_path)
        generator.set_state_dict(checkpoint['generator'])

        kp_detector.set_state_dict(checkpoint['kp_detector'])

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

    def make_animation(self,
                       source_image,
                       driving_video,
                       generator,
                       kp_detector,
                       relative=True,
                       adapt_movement_scale=True):
        with paddle.no_grad():
      

            
            predictions = []
            source = paddle.to_tensor(source_image[np.newaxis].astype(
                np.float32)).transpose([0, 3, 1, 2])
            kp_source = kp_detector(source)
            kp_source_batch = {}
            kp_source_batch["value"] = paddle.tile(kp_source["value"], repeat_times=[self.batch_size, 1, 1])
            kp_source_batch["jacobian"] = paddle.tile(kp_source["jacobian"], repeat_times=[self.batch_size, 1, 1, 1])
            source = paddle.tile(source, repeat_times=[self.batch_size, 1, 1, 1])

            driving = paddle.to_tensor(
                np.array(driving_video[:1]).astype(
                    np.float32)).transpose([0, 3, 1, 2])
            kp_driving_initial = kp_detector(driving[:1])
            
            begin_idx = 0
            for _ in tqdm(range(int(np.ceil(float(len(driving_video)) / self.batch_size)))):
                frame_num = min(self.batch_size, len(driving_video) - begin_idx)
                driving = paddle.to_tensor(
                    np.array(driving_video[begin_idx:begin_idx + frame_num]).astype(
                    np.float32)).transpose([0, 3, 1, 2])

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
                    adapt_movement_scale=adapt_movement_scale)

                out = generator(source[0:frame_num], kp_source=kp_source_img, kp_driving=kp_norm)
                img = np.transpose(out['prediction'].numpy(), [0, 2, 3, 1]) * 255.0
                if self.face_enhancement:
                #     _, _, img = self.faceenhancer.enhance(img[0])
                    img = self.faceenhancer.enhance_from_batch(img)
                # print(img.shape)
                predictions.append(img)
                begin_idx += frame_num
        return np.concatenate(predictions)

    def extract_bbox(self, image):
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            face_detector=self.face_detector)

        # frame = [image]
        predictions = detector.get_detections_for_image(np.array(image))
        result, coords = self.detection_func(image, predictions)
        return np.array(result), np.array(coords)


    def extract_masks(self, results, coords, source_image, model_type="face_seg"):
        if len(results) == 1:
            return 
        if model_type == "face_seg": 
            face_model = FaceSeg()
        else:
            face_model = FaceParser()
        box_masks = []
        frame = source_image.copy()     
        polygons = [polygon2ellipsemask(coord, frame.shape[:2]) for coord in coords]
        for i in tqdm(range(0, len(results), 2)):
            x1, y1, x2, y2, _ = results[i]['rec']
            if i + 1 < len(results):
                x11, y11, x21, y21, _ = results[i+1]['rec']
                width, height = max(x2-x1, x21-x11), max(y2-y1, y21-y11)
                mask_image = cv2.hconcat([cv2.resize(frame[y1:y2, x1:x2], (width, height)),
                                            cv2.resize(frame[y11:y21, x11:x21], (width, height))])
                h, w = mask_image.shape[:2]
                if model_type == "face_parser":
                    box_mask = face_model.parse(cv2.resize(mask_image, (512, 512)).astype(np.float32))
                    box_mask = cv2.resize(np.array(box_mask).astype('uint8'), (w, h))        
                    box_mask[box_mask != 0] = 1
                else:
                    box_mask = cv2.resize(face_model(mask_image), (w, h))
                ### masks + detections
                # box_masks.append(cv2.bitwise_and(cv2.resize(box_mask[:, :w//2], (x2-x1, y2-y1)), 
                #                                polygon2mask(coords[i], frame.shape[:2])[y1:y2, x1:x2]))
                # box_masks.append(cv2.bitwise_and(cv2.resize(box_mask[:, w//2:], (x21-x11, y21-y11)), 
                #                                polygon2mask(coords[i+1], frame.shape[:2])[y11:y21, x11:x21]))
                ### masks + ellipse detections
                box_masks.append(cv2.bitwise_and(cv2.resize(box_mask[:, :w//2], (x2-x1, y2-y1)), 
                                              polygons[i][y1:y2, x1:x2]))
                box_masks.append(cv2.bitwise_and(cv2.resize(box_mask[:, w//2:], (x21-x11, y21-y11)), 
                                              polygons[i+1][y11:y21, x11:x21]))
                
                ### just ellipse detections
                # box_masks.append(polygons[i][y1:y2, x1:x2])
                # box_masks.append(polygons[i+1][y11:y21, x11:x21])
            else:
                if model_type == "face_parser":
                    box_mask = face_model.parse(cv2.resize(frame[y1:y2, x1:x2], (512, 512)).astype(np.float32))
                    box_mask = cv2.resize(np.array(box_mask).astype('uint8'), (x2-x1, y2-y1))        
                    box_mask[box_mask != 0] = 1
                else:
                    box_mask = face_model(frame[y1:y2, x1:x2])
                ### just ellipse detections
                # box_masks.append(polygons[i][y1:y2, x1:x2])
                ### masks + ellipse detections
                box_masks.append(cv2.bitwise_and(box_mask, 
                                               polygons[i][y1:y2, x1:x2]))
                ### masks + detections
                # box_masks.append(cv2.bitwise_and(cv2.resize(box_mask, (x2-x1, y2-y1)), 
                #                                polygon2mask(coords[i], frame.shape[:2])[y1:y2, x1:x2]))
        return box_masks


def scale_bboxes(img1_shape, bboxes, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    bboxes[:, [0, 2]] -= pad[0]  # x padding
    bboxes[:, [1, 3]] -= pad[1]  # y padding
    bboxes[:, :4] /= gain

    # clip coords
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, img0_shape[0])
    return bboxes


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[:,] /= gain

    # clip coords
    coords[:, 0] = coords[:, 0].clip(0, img0_shape[1])  # x1, x2
    coords[:, 1] = coords[:, 1].clip(0, img0_shape[0])
    return coords