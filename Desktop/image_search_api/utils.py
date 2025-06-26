# utils.py
from PIL import Image
import numpy as np
import torch
import io

def preprocess_image(image_file, processor):
    image = Image.open(io.BytesIO(image_file)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)

def extract_feature(pixel_values, model, device):
    with torch.no_grad():
        embedding = model(pixel_values.unsqueeze(0).to(device)).last_hidden_state[:, 0, :]
    return embedding.squeeze(0).cpu().numpy().astype("float32")

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)
