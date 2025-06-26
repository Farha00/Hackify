import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

# Set the device to CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet-50 model and replace the final layer
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Modify the last layer for your task
n_inputs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(n_inputs, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 4),
    nn.LogSigmoid(),
)

# Explicitly load the state dict with map_location to ensure compatibility with CPU
model_path = 'models/bt_resnet50_model.pt'
resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# Set the model to evaluation mode and move it to the correct device
resnet_model.eval()
resnet_model.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Label names for the output
LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']

# Get the image path from user input
img_name = input("Enter the path to the image: ")

# Check if the provided image path is valid
if not os.path.exists(img_name):
    print("File does not exist. Please provide a valid path.")
    sys.exit(1)

# Load and preprocess the image
img = Image.open(img_name)
img = transform(img)

# Add batch dimension and move the image to the correct device
img = img.unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    y_hat = resnet_model(img)
    predicted = torch.argmax(y_hat, dim=1)

# Output the predicted class
predicted_class = LABELS[predicted.item()]
print(f"Predicted class: {predicted_class}")
