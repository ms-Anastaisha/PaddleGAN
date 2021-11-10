import sys
sys.path.insert(0, '/home/anastasia/paddleGan/PaddleGAN/')
import os 
from ppgan.apps.first_order_predictor import FirstOrderPredictor
import time
args = {
    "output": "output",
    "filename": "result.mp4",
    "weight_path": None,
    "relative": True,
    "adapt_scale": False,
    "find_best_frame": False,
    "best_frame": None,
    "ratio": 0.4,
    "face_detector": "sfd",
    "multi_person":  True,
    "image_size": 256,
    "batch_size": 64,
    "face_enhancement": False,
    "gfpgan_model_path": "/home/anastasia/paddleGan/PaddleGAN/GFPGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
    "mobile_net": False,
    "preprocessing": True,
    "face_align": False,
}
resources = {
    "source_image": [#"/home/user/paddle/PaddleGAN/data/selfie.jpeg", 
                    "/home/anastasia/paddleGan/PaddleGAN/data/selfie.jpeg"],
                    #"/home/user/paddle/PaddleGAN/data/selfie4.jpg", 
                    #"/home/user/paddle/PaddleGAN/data/selfie5.jpg"],
    "driving_video": "/home/anastasia/paddleGan/PaddleGAN/data/Jingle Bells A Cappella Backtrack_1024.mp4"
}
if __name__ == '__main__':
    start = time.time()
    predictor = FirstOrderPredictor(**args)
    
    for img_path in resources["source_image"]:
        basename = os.path.basename(img_path) 
        name, ext = os.path.splitext(basename)

        predictor.run(img_path, resources["driving_video"], name + "16sec ellipse" + '.mp4')
    print("inference time (for 16 sec video):", (time.time() - start) / len(resources["source_image"]))