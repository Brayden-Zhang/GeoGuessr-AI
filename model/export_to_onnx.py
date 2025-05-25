import torch
from vit_model import create_vit_model
import os
import json
import glob

def get_class_names():
    # Get all country directories from the dataset
    dataset_path = os.path.join("compressed_dataset")
    country_dirs = sorted([d for d in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(d)])
    return [os.path.basename(d) for d in country_dirs]

def export_to_onnx():
    try:
        # Load the PyTorch model
        checkpoint = torch.load('model/checkpoints/best_vit_model.pth', map_location='cpu')
        
        # Create model with the same number of classes
        num_classes = checkpoint['model_state_dict']['classifier.4.weight'].shape[0]
        model = create_vit_model(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create the extension/model directory if it doesn't exist
        os.makedirs('extension/model', exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            'extension/model/model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=12
        )

        # Get class names from dataset
        class_names = get_class_names()
        
        # Save class names
        with open('extension/model/class_names.json', 'w') as f:
            json.dump(class_names, f)

        print("Model exported successfully to ONNX format")
        print(f"Number of classes: {num_classes}")
        print(f"Class names saved: {len(class_names)} countries")
        
        return True
    except Exception as e:
        print(f"Error exporting model: {str(e)}")
        return False

if __name__ == '__main__':
    export_to_onnx() 