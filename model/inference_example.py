import os
import torch
from torchvision import transforms
from PIL import Image
from model import CNNModel

# Paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(root_dir, 'image_samples', '9983.png')
model_path = os.path.join(root_dir, 'model/checkpoints', 'best_model.pth')

# Check if files exist
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = CNNModel().to(device)

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Extract and load the model state_dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
        print("Model loaded successfully from 'model_state_dict'.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
else:
    try:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully from checkpoint.")
    except RuntimeError as e:
        print(f"Error loading checkpoint directly: {e}")
        raise

model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Ensure this matches the model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_coordinates(image):
    image = transform(image).unsqueeze(0).to(device)
    print("Input tensor shape:", image.shape)  # Debugging
    with torch.no_grad():
        output = model(image)
    return output.cpu().numpy()

def infer_from_file(file_path):
    try:
        # Load the image
        image = Image.open(file_path).convert('RGB')
        
        result = predict_coordinates(image)
        return result.tolist()
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    result = infer_from_file(image_path)
    print("Inference result:", result)
