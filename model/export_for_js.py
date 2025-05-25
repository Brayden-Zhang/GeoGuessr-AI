import torch
from vit_model import create_vit_model
import os

def export_model_for_js():
    # Load the PyTorch model
    checkpoint = torch.load('model/checkpoints/best_vit_model.pth', map_location='cpu')
    model = create_vit_model(num_classes=checkpoint['model_state_dict']['classifier.weight'].shape[0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create the extension/model directory if it doesn't exist
    os.makedirs('extension/model', exist_ok=True)

    # Export the model to TorchScript format
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the model
    traced_model.save('extension/model/model.pt')

    # Save the class names
    import json
    class_names = checkpoint.get('class_names', [])  # Get class names from checkpoint if available
    with open('extension/model/class_names.json', 'w') as f:
        json.dump(class_names, f)

if __name__ == '__main__':
    export_model_for_js() 