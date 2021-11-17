import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import gdown
from pathlib import Path


cur_path = os.path.abspath(os.path.dirname(__file__))
FairFace_weights = os.path.join(cur_path, "./res34_fair_align_multi_7_20190809.pt")
FairFace_weights_url = "https://drive.google.com/uc?id=1QLDvwj6kCGZIvKF3qGausoOLfE_X3usT"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FaceClassification():
    def __init__(self, weights: str = FairFace_weights):
        super(FaceClassification, self).__init__()

        self.weights = self.check_weights(weights)
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(torch.load(FairFace_weights))
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def classify_image(self, image: np.ndarray):
        # TODO: check how face alignment affects the quality of classification

        image = self.transform(image)
        # reshape image to match model dimensions (1 batch size)
        image = image.view(1, 3, 224, 224)
        image = image.to(device)

        outputs = self.model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        # Genders
        # 0 - Male
        # 1 - Female

        # Ages
        # 0 - '0-2'
        # 1 - '3-9'
        # 2 - '10-19'
        # 3 - '20-29'
        # 4 - '30-39'
        # 5 - '40-49'
        # 6 - '50-59'
        # 7 - '60-69'
        # 8 - '70+'

        return gender_pred, age_pred

    def check_weights(self, weights: str):
        if Path(weights).exists():
            return weights
        else:
            gdown.download(
                FairFace_weights_url,
                FairFace_weights,
                quiet=False
            )
            return FairFace_weights
