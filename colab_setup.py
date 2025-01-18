import os
import yaml
import kagglehub
from google.colab import drive

def setup_colab():
    print("Mounting Google Drive...")
    drive.mount('/content/drive')

    print("Installing dependencies...")
    os.system('pip install torch torchvision tqdm tensorboard kagglehub pandas')

    print("Cloning repository...")
    token = os.getenv('GITHUB_TOKEN')  # Make sure to set this in Colab
    os.system(f'git clone https://brayden-zhang:{token}@github.com/Brayden-Zhang/GeoGuessr-AI.git')
    os.chdir('GeoGuessr-AI')

    print("Downloading dataset...")
    dataset_path = "/content/dataset"
    os.makedirs(dataset_path, exist_ok=True)
    
    kagglehub.init()
    kagglehub.download_dataset('paulchambaz/google-street-view', dataset_path)
    print(f"Dataset downloaded to: {dataset_path}")

    print("Updating configuration...")
    config_path = 'cfgs/train_cfgs.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset']['data_path'] = dataset_path
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("Starting training...")
    os.system('python train.py')

setup_colab() 