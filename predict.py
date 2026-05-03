# predict.py
import torch
from torchvision import transforms
from PIL import Image
import config
from model import DeepfakeDetectorCNN

def classify_images(list_of_img_paths, model_weights_path='_checkpoints/_final_weights.pth'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize model and load trained weights
    model = DeepfakeDetectorCNN()
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Transformation pipeline for inference
    data_transforms = transforms.Compose([
        transforms.Resize((config.resize_x, config.resize_y)),
        transforms.ToTensor()
    ])
    
    labels = []
    
    with torch.no_grad():
        for img_path in list_of_img_paths:
            img = Image.open(img_path).convert('RGB')
            # Add a batch dimension of 1 since PyTorch expects [Batch, Channels, Height, Width]
            img_tensor = data_transforms(img).unsqueeze(0).to(device) 
            
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Since 'fake' is 0 and 'real' is 1
            label = "real" if predicted.item() == 1 else "fake"
            labels.append(label)
            
    return labels