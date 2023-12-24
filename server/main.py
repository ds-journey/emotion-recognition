import base64
import json
import os
import random
import secrets

import cv2
from fastapi import FastAPI, WebSocket
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2 as v2_transforms
from torchvision.models import efficientnet
from torchvision.utils import save_image
import torchvision.models as tv_models
import time


model_version = 2
epoch = 30
normalize = True
start = time.monotonic()
prev_save = start


app = FastAPI()


MODEL_FILE = os.getenv('MODEL_FILE')
IMAGES_DIR = os.getenv('IMAGES_DIR')

if not MODEL_FILE:
    raise NotImplementedError(f"MODEL_FILE environment variable must be set.")

if MODEL_FILE == 'default':
    MODEL_FILE = os.path.expanduser(f'~/data/torch/model_{model_version}_{epoch}.th')

if IMAGES_DIR == 'default':
    IMAGES_DIR = os.path.expanduser(f'~/tmp')

if not os.path.isfile(MODEL_FILE):
    raise NotImplementedError(f"MODEL_FILE path '{MODEL_FILE}' does not exist or is not a file.")


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for pytorch')


class Model(nn.Module):
    def __init__(self, model_version: int = 1, dropout: float = 0.975):
        super().__init__()
        num_classes = 6
        self.model_version = model_version
        model_factories = [tv_models.efficientnet_v2_s, tv_models.efficientnet_v2_m, tv_models.efficientnet_v2_l]
        model_weights = [efficientnet.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
                         efficientnet.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
                         efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1]
        self.base = model_factories[model_version-1](weights=model_weights[model_version-1])
        self.freeze_params()
        self.base.classifier[0] = nn.Dropout(dropout, inplace=True)
        self.base.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        self.base.classifier.add_module('2', nn.Softmax(dim=1))
        self.to(device)

    def forward(self, xb):
        return self.base(xb)

    def freeze_params(self):
        return


model = Model(model_version=model_version)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()


mean = [0.5432, 0.4287, 0.3821]
std = [0.2862, 0.2541, 0.2465]
IMAGE_SIZE = (224, 224)
norm_transforms = [v2_transforms.Normalize(mean=mean, std=std)] if normalize else []
basic_transforms = [
    v2_transforms.ToImage(),
    v2_transforms.ToDtype(torch.float32, scale=True),
    v2_transforms.Resize(IMAGE_SIZE, antialias=True),
]
working_transforms = v2_transforms.Compose(basic_transforms + norm_transforms)


emotions = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    with torch.no_grad():
        await websocket.accept()
        while True:
            payload = await websocket.receive_text()
            payload = json.loads(payload)
            imageByt64 = payload['data']['image'].split(',')[1]

            # decode and convert into image
            image = np.fromstring(base64.b64decode(imageByt64), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = torch.from_numpy(image)
            image = torch.permute(image, (2, 0, 1))
            # apply image transforms
            image = working_transforms(image).to(device)
            image = image.unsqueeze(0)

            persist_image(image)

            # Detect Emotion via Tensorflow model
            out = model(image)
            out = out.squeeze()
            # idx = torch.argmax(out, dim=0); emotion = emotions[idx]

            response = {
                "predictions": [round(i, 5) for i in out.tolist()],
            }
            await websocket.send_json(response)


def persist_image(img):
    if not IMAGES_DIR or not os.path.isdir(IMAGES_DIR):
        return
    global prev_save
    if time.monotonic() - prev_save < 10:
        return
    prev_save = time.monotonic()
    mean_t = torch.tensor(mean).detach().to(device).view(-1, 1, 1) if normalize else 0
    std_t = torch.tensor(std).detach().to(device).view(-1, 1, 1) if normalize else 1
    img = img * std_t
    img = img + mean_t
    save_image(img.squeeze(), os.path.join(IMAGES_DIR, secrets.token_hex(1) + '.png'))