#!/usr/bin/python3
from io import BytesIO
from PIL import Image
import requests
import clip
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

if __name__ == '__main__':
    for url in sys.stdin:
        response = requests.get(url.strip())
        response.raise_for_status()
        image = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0).to(device)
        with torch.no_grad():
            print(model.encode_image(image)[0].tolist())
            sys.stdout.flush()
