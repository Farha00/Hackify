# model.py
import torch
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINOv2 model and processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-large")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

# Load FAISS index and paths
doc_features = np.load("data/doc_features.npy").astype("float32")
doc_paths = np.load("data/doc_paths.npy", allow_pickle=True)


index = faiss.IndexFlatL2(doc_features.shape[1])
index.add(doc_features)
