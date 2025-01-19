import os
import sys
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.append(root_dir)


from model import CNNModel

# Device setup for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the CNNModel
@st.cache_resource
def load_model():
    try:
        model = CNNModel().to(device)
        model.eval()  # Set the model to evaluation mode
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load the CNNModel: {e}")
        sys.exit(1)

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to match model input size
        transforms.ToTensor(),         # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Run inference on the model
def run_inference(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        return outputs

# Streamlit app layout
st.title("GeoGuessr AI Inference")
st.write("Upload an image, and the CNNModel will process it for inference.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Preprocessing the image...")
    image_tensor = preprocess_image(image)

    # Run inference
    st.write("Running model inference...")
    outputs = run_inference(model, image_tensor)

    # Display inference results
    st.write("Model Output:")
    st.json(outputs.cpu().numpy().tolist())
