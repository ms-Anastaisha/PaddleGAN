import sys
sys.path.insert(0, '/home/anastasia/paddleGan/PaddleGAN/')
import os 
from ppgan.apps.first_order_predictor import FirstOrderPredictor
import time
import pathlib
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
    "multi_person": True,
    "image_size": 256,
    "batch_size": 30,
    "face_enhancement": False,
    "gfpgan_model_path": None, #"/home/user/paddle/PaddleGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
    "mobile_net": False,
    "preprocessing": True,
    "solov_path": "/home/anastasia/paddleGan/PaddleGAN/PaddleDetection/solov2_r50_enhance_coco"
}
resources = {
    "source_image": "/home/anastasia/paddleGan/PaddleGAN/data/source_image",
                    #[#"/home/user/paddle/PaddleGAN/data/selfie.jpeg", 
                    #"/home/anastasia/paddleGan/PaddleGAN/data/source_image/download-21.png"],
                    #"/home/user/paddle/PaddleGAN/data/selfie4.jpg", 
                    #"/home/user/paddle/PaddleGAN/data/selfie5.jpg"],
    "driving_video": ["/home/anastasia/paddleGan/PaddleGAN/data/video16.mp4",
    # "/home/anastasia/paddleGan/PaddleGAN/data/mayiyahei.MP4", 
    # "/home/anastasia/paddleGan/PaddleGAN/data/jingle_bells_acapella_part.mp4"
    ], 
    "audio": None, 
    "borders": {"landscape": "/home/anastasia/paddleGan/PaddleGAN/data/borders/landscape_border.png",
                "square": "/home/anastasia/paddleGan/PaddleGAN/data/borders/square_border.png", 
                "portrait": "/home/anastasia/paddleGan/PaddleGAN/data/borders/portrait_border.png"}
}
if __name__ == '__main__':
    start = time.time()
    predictor = FirstOrderPredictor(**args)
    resources["source_image"] = [str(filepath.absolute()) for filepath in pathlib.Path(resources["source_image"]).glob('**/*')]
    
    for img_path in resources["source_image"]:
        basename = os.path.basename(img_path) 
        name, ext = os.path.splitext(basename)

        predictor.run(img_path, resources["driving_video"], name  + '.mp4', None, resources["borders"])
    print("inference time (for 16 sec video):", (time.time() - start) / len(resources["source_image"]))