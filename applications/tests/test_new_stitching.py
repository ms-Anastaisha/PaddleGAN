import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.insert(0, os.path.dirname(root_path))

from ppgan.apps.first_order_predictor import FirstOrderPredictor
import time

args = {
    "output": "output",
    "filename": "result.mp4",
    "weight_path": None,
    "relative": True,
    "adapt_scale": True,
    "find_best_frame": False,
    "best_frame": None,
    "ratio": 0.4,
    "face_detector": "sfd",
    "multi_person":  True,
    "image_size": 256,
    "batch_size": 16,
    "face_enhancement": False,
    "gfpgan_model_path": None, #"/home/user/paddle/PaddleGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
    "mobile_net": False,
    "preprocessing": True,
    "solov_path": "/home/user/paddle/PaddleGAN/PaddleDetection/solov2_r50_enhance_coco"
}

resources = {
    "source_image": [#"/home/user/paddle/PaddleGAN/data/selfie.jpeg", 
                    "/home/user/paddle/PaddleGAN/data/selfie2.JPEG"],
                    #"/home/user/paddle/PaddleGAN/data/selfie4.jpg", 
                    #"/home/user/paddle/PaddleGAN/data/selfie5.jpg"],
    #"driving_video": "/home/user/paddle/PaddleGAN/data/vox_example.mp4",
    "driving_video": [
        "/home/user/paddle/PaddleGAN/data/Remini_Animation (1).mp4",
        "/home/user/paddle/PaddleGAN/data/video10.mp4",
        "/home/user/paddle/PaddleGAN/data/jingle_bells_acapella_part.mp4", 

     ],
    "audio": "/home/user/paddle/PaddleGAN/data/just_audio.mp4"
}


if __name__ == '__main__':
    start = time.time()
    predictor = FirstOrderPredictor(**args)
    
    for img_path in resources["source_image"]:
        basename = os.path.basename(img_path) 
        name, ext = os.path.splitext(basename)

       
        predictor.run(img_path, resources["driving_video"], name + "16sec" + '.mp4', resources["audio"])
    print("inference time (for 16 sec video):", (time.time() - start) / len(resources["source_image"]))
