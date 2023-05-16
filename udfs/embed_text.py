#!/usr/bin/python3
import clip
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

if __name__ == '__main__':
    for text in sys.stdin:
        inputs = clip.tokenize(text)
        with torch.no_grad():
            text_features = []
            text_features = model.encode_text(inputs)[0].tolist()
            print(text_features)
            sys.stdout.flush()