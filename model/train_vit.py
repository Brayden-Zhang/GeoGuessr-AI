import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
from .vit_model import create_vit_model
from .country_dataset import CountryDataset
from .get_images import path
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

def plot_confusion_matrix(cm, class_names, writer, epoch):
    """Plot confusion matrix and save to TensorBoard"""
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save to TensorBoard
    writer.add_figure('Confusion Matrix', plt.gcf(), epoch)
    plt.close()

def train_vit_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the Vision Transformer model
    
    Args:
        model: The ViT model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to train on
    """
    # Ensure CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = 'cpu'
    elif device == 'cuda':
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Set CUDA device properties for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Create TensorBoard writer
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'vit_training_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('model/checkpoints', exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Enable mixed precision training
    
    # Log model graph (ensure model and input are on same device)
    try:
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"Warning: Could not create model graph: {str(e)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    best_val_acc = 0.0
    class_names = val_loader.dataset.dataset.get_class_names()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Regular training
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:  # Log every 10 batches
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Accuracy/train_batch', 100. * train_correct / train_total, 
                                epoch * len(train_loader) + batch_idx)
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels for classification report
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Log epoch metrics
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
        writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Print classification report every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\nClassification Report:")
            # Get unique classes present in the validation data
            unique_classes = np.unique(all_labels)
            present_class_names = [class_names[i] for i in unique_classes]
            
            report = classification_report(all_labels, all_preds, 
                                        target_names=present_class_names, 
                                        output_dict=True)
            print(classification_report(all_labels, all_preds, 
                                     target_names=present_class_names))
            
            # Log per-class metrics
            for class_name in present_class_names:
                writer.add_scalar(f'Precision/{class_name}', report[class_name]['precision'], epoch)
                writer.add_scalar(f'Recall/{class_name}', report[class_name]['recall'], epoch)
                writer.add_scalar(f'F1-score/{class_name}', report[class_name]['f1-score'], epoch)
            
            # Log confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(cm, present_class_names, writer, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'model/checkpoints/best_vit_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    writer.close()

def main():
    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset_path = os.path.join(path, "compressed_dataset")
    full_dataset = CountryDataset(dataset_path, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Increased batch size
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,  # Increased batch size
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model
    num_classes = full_dataset.get_num_classes()
    model = create_vit_model(num_classes=num_classes, pretrained=True)
    
    # Train model
    train_vit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=1e-4,
        weight_decay=0.01
    )

if __name__ == '__main__':
    main() 