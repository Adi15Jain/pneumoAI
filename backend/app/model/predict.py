# backend/app/model/predict.py

from PIL import Image
import torch
from torchvision import transforms
from .cnn import PneumoniaCNN

device = torch.device("cpu")

model_path = "/Users/adijain/Documents/Projects/pneumonia_detection/backend/saved_models/pneumonia_cnn.pth"

model = PneumoniaCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ✅ Updated transform with .convert("L") for grayscale
def preprocess_image(image_bytes):
    image_bytes.seek(0)  # ✅ Important for reading correctly
    image = Image.open(image_bytes).convert("L")  # ✅ Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Grayscale -> shape [1, 224, 224]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(image).unsqueeze(0)  # Shape: [1, 1, 224, 224]
    return tensor


def predict_image(image_bytes):
    tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.sigmoid(output).item()
    return prediction
