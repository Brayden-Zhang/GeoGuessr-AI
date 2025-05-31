import base64
import io

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# Import custom modules
from model.country_dataset import CountryDataset
from model.vit_model import ViTModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Global Variables ---
model = None
class_names = None
device = None
num_classes = None


def load_model_and_class_names():
    global model, class_names, device, num_classes

    # Initialize CountryDataset to get num_classes and class_names
    # Assuming 'model/compressed_dataset' is the correct path relative to the repo root
    dataset = CountryDataset(root_dir='model/compressed_dataset')
    num_classes = dataset.get_num_classes()
    class_names = dataset.get_class_names()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ViTModel
    model = ViTModel(num_classes=num_classes)

    # Load checkpoint
    # Ensure the checkpoint path is correct, relative to the repo root
    checkpoint_path = 'model/checkpoints/best_vit_model.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Adjust how state_dict is loaded based on how it was saved in the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        # Handle the error appropriately, e.g., exit or use a default model
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}.")
    print(f"Number of classes: {num_classes}")
    # print(f"Class names: {class_names}") # Potentially very long list


# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    """
    Preprocesses image bytes into a tensor for model inference.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Standard ImageNet normalization
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None


@app.route("/")
def hello():
    return "GeoGuessr AI Backend is running!"


# --- API Endpoints ---
@app.route("/predict", methods=["POST"])
def predict():
    global model, class_names, device

    if model is None or class_names is None or device is None:
        return jsonify({"error": "Model not loaded or not ready"}), 500

    if not request.json or 'image_data' not in request.json:
        return jsonify({"error": "Missing 'image_data' in JSON payload"}), 400

    try:
        image_data_base64 = request.json['image_data']
        # Remove potential header like "data:image/jpeg;base64,"
        if "," in image_data_base64:
            image_data_base64 = image_data_base64.split(',')[1]

        image_bytes = base64.b64decode(image_data_base64)
    except Exception as e:
        return jsonify({"error": f"Error decoding base64 image data: {e}"}), 400

    image_tensor = preprocess_image(image_bytes)
    if image_tensor is None:
        return jsonify({"error": "Image preprocessing failed"}), 500

    image_tensor = image_tensor.to(device)

    try:
        with torch.no_grad():
            output = model(image_tensor)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        country_name = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        return jsonify({
            "prediction": country_name,
            "confidence": confidence_score
        })

    except Exception as e:
        print(f"Error during model inference: {e}")
        return jsonify({"error": f"Model inference failed: {e}"}), 500


# --- Main Block ---
if __name__ == "__main__":
    load_model_and_class_names()
    if model is not None: # Only run app if model loaded successfully
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("Failed to load model. Flask app will not start.")
