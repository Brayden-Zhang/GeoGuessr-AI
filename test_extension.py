import os
import shutil
import subprocess

def test_extension_setup():
    # Check if model files exist
    model_files = [
        'model/checkpoints/best_vit_model.pth',
        'model/vit_model.py'
    ]
    
    for file in model_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found")
            return False
    
    # Export model to ONNX
    print("Exporting model to ONNX format...")
    result = subprocess.run(['python', 'model/export_to_onnx.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error exporting model:")
        print(result.stderr)
        return False
    print(result.stdout)
    
    # Check if extension files exist
    extension_files = [
        'extension/manifest.json',
        'extension/popup.html',
        'extension/popup.js',
        'extension/content.js',
        'extension/background.js',
        'extension/model/model.onnx',
        'extension/model/class_names.json'
    ]
    
    for file in extension_files:
        if not os.path.exists(file):
            print(f"Error: Extension file {file} not found")
            return False
    
    print("\nExtension setup completed successfully!")
    print("\nTo load the extension in Chrome:")
    print("1. Open Chrome and go to chrome://extensions/")
    print("2. Enable 'Developer mode' in the top right")
    print("3. Click 'Load unpacked' and select the 'extension' directory")
    print("\nThe extension should now be ready to use!")
    
    return True

if __name__ == '__main__':
    test_extension_setup() 