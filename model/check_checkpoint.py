import torch

def inspect_checkpoint():
    checkpoint = torch.load('model/checkpoints/best_vit_model.pth', map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())
    if 'model_state_dict' in checkpoint:
        print("\nModel state dict keys:", checkpoint['model_state_dict'].keys())
        # Print the first few keys and their shapes
        for key in list(checkpoint['model_state_dict'].keys())[:5]:
            print(f"{key}: {checkpoint['model_state_dict'][key].shape}")

if __name__ == '__main__':
    inspect_checkpoint() 