import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNNModel
from dataset import GeoGuessrDataset
from dataset import train_dataset, val_dataset, test_dataset
from tqdm import tqdm


def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc='Training')
        for images, coordinates in train_bar:
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coordinates)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for images, coordinates in val_bar:
                images = images.to(device)
                coordinates = coordinates.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, coordinates)
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print epoch statistics
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved new best model!')
            
        print('-' * 50)

# Usage example
if __name__ == "__main__":
    model = CNNModel()
    
# Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    train_model(model, train_loader, val_loader) 