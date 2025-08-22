#!/usr/bin/env python3
"""
PETer Model Training Script Based on Paper Method
Training using sit_data, squat_data, stand_data three groups of data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from peter_network import PETerNetwork, DataProcessor
import warnings
warnings.filterwarnings('ignore')

class RadarPointCloudDataset(Dataset):
    """Radar point cloud dataset class"""
    
    def __init__(self, data_paths, labels, sample_indices, num_points=100, num_frames=25, transform=None, augment=False):
        """
        Initialize dataset
        Args:
            data_paths: List of data file paths
            labels: Corresponding label list
            sample_indices: Sample index list
            num_points: Number of points sampled per frame
            num_frames: Number of frames per sample
            transform: Data transformation
            augment: Whether to perform data augmentation
        """
        self.data_paths = data_paths
        self.labels = labels
        self.sample_indices = sample_indices
        self.num_points = num_points
        self.num_frames = num_frames
        self.transform = transform
        self.augment = augment
        
        # Label mapping
        self.label_to_idx = {'sit': 0, 'squat': 1, 'stand': 2}
        self.idx_to_label = {0: 'sit', 1: 'squat', 2: 'stand'}
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # Load data
        with open(self.data_paths[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get sample at specified index
        sample_idx = self.sample_indices[idx]
        sample = data[sample_idx]
        
        # Get target point cloud data
        target_points = sample['target_points']
        if len(target_points) == 0:
            # If no target points, create zero-padded data
            target_points = np.zeros((self.num_points, 3))
        else:
            target_points = np.array(target_points)
        
        # Normalize point cloud
        target_points = self.normalize_point_cloud(target_points)
        
        # Sample fixed number of points
        target_points = self.sample_points(target_points, self.num_points)
        
        # Data augmentation (only for training set)
        if self.augment:
            target_points = self.augment_point_cloud(target_points)
        
        # Add intensity information (set to 1 if not present)
        intensity = np.ones((len(target_points), 1))
        target_points = np.hstack([target_points, intensity])
        
        # Create time sequence (simplified to single frame here, should have multiple frames in practice)
        # To adapt to the model, we repeat single frame data
        sequence = np.tile(target_points, (self.num_frames, 1, 1))
        
        # Convert to tensor
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([self.label_to_idx[self.labels[idx]]])
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label
    
    def normalize_point_cloud(self, points):
        """Normalize point cloud"""
        if len(points) == 0:
            return points
        
        # Center
        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] = points[:, :3] - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
        if max_dist > 0:
            points[:, :3] = points[:, :3] / max_dist
        
        return points
    
    def sample_points(self, points, num_points):
        """Fixed point sampling"""
        if len(points) == 0:
            return np.zeros((num_points, 3))
        
        if len(points) >= num_points:
            # Random sampling
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices]
        else:
            # Repeat sampling
            indices = np.random.choice(len(points), num_points, replace=True)
            return points[indices]
    
    def augment_point_cloud(self, points):
        """Point cloud data augmentation"""
        if len(points) == 0:
            return points
        
        # Random rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-0.1, 0.1)  # Small angle rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            points[:, :3] = points[:, :3] @ rotation_matrix.T
        
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, points[:, :3].shape)
            points[:, :3] += noise
        
        # Random scaling
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale
        
        return points

def load_training_data():
    """Load training data"""
    print("Loading training data...")
    
    data_paths = []
    labels = []
    sample_indices = []
    
    # Data directory configuration
    data_config = {
        'sit': 'radar_data/sit_improved_fused/improved_fused_pointcloud',
        'squat': 'radar_data/squat_improved_fused/improved_fused_pointcloud', 
        'stand': 'radar_data/stand_improved_fused/improved_fused_pointcloud'
    }
    
    for action, data_dir in data_config.items():
        if os.path.exists(data_dir):
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            print(f"   {action}: {len(json_files)} files")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        # Each sample as a data point
                        for i, sample in enumerate(data):
                            if isinstance(sample, dict) and 'target_points' in sample:
                                data_paths.append(json_file)
                                labels.append(action)
                                sample_indices.append(i)
                except Exception as e:
                    print(f"   Failed to read {json_file}: {e}")
    
    print(f"Total data points: {len(data_paths)}")
    
    # Count label distribution
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")
    
    return data_paths, labels, sample_indices

def create_data_loaders(data_paths, labels, sample_indices, batch_size=32, test_size=0.2, val_size=0.1):
    """Create data loaders"""
    print("Creating data loaders...")
    
    # Split into training, validation, and test sets
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        data_paths, labels, sample_indices, test_size=test_size, stratify=labels, random_state=42
    )
    
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=val_size, stratify=y_temp, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets (training set uses data augmentation)
    train_dataset = RadarPointCloudDataset(X_train, y_train, idx_train, augment=True)
    val_dataset = RadarPointCloudDataset(X_val, y_val, idx_val, augment=False)
    test_dataset = RadarPointCloudDataset(X_test, y_test, idx_test, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, y_train

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cpu', class_weights=None):
    """Train model"""
    print("Starting model training...")
    
    # Move to device
    model = model.to(device)
    
    # Loss function and optimizer
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss function, weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard loss function")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate accuracy
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_peter_model.pth')
            print(f'  Saved best model (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping training (Patience: {patience})')
                break
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model"""
    print("Evaluating model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    print(f'Test accuracy: {accuracy:.2f}%')
    
    # Classification report
    print('\nClassification report:')
    print(classification_report(all_targets, all_predictions, 
                               target_names=['sit', 'squat', 'stand']))
    
    return all_predictions, all_targets, accuracy

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_class_weights(labels):
    """Calculate class weights to handle data imbalance"""
    from collections import Counter
    
    # Count samples for each class
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights (fewer samples, higher weight)
    weights = []
    for i in range(3):  # 3 classes: sit, squat, stand
        class_name = ['sit', 'squat', 'stand'][i]
        if class_name in class_counts:
            # Use inverse frequency weights
            weight = total_samples / (len(class_counts) * class_counts[class_name])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    print(f"Class weights: {weights}")
    return weights

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['sit', 'squat', 'stand'],
                yticklabels=['sit', 'squat', 'stand'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_training_summary(class_weights, accuracy):
    """Print training summary"""
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Class weights: {class_weights}")
    print(f"Final test accuracy: {accuracy:.2f}%")
    print("Techniques used:")
    print("   Weighted loss function (handle data imbalance)")
    print("   Data augmentation (rotation, noise, scaling)")
    print("   Early stopping (prevent overfitting)")
    print("   Learning rate scheduling (optimize convergence)")
    print("   PETer network architecture (EdgeConv + Transformer)")
    print("="*60)

def main():
    """Main function"""
    print("PETer Model Training Started")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_paths, labels, sample_indices = load_training_data()
    
    if len(data_paths) == 0:
        print("No valid data found")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader, y_train = create_data_loaders(data_paths, labels, sample_indices)
    
    # Calculate class weights to handle data imbalance
    class_weights = calculate_class_weights(y_train)
    
    # Create model
    model = PETerNetwork(num_classes=3, num_points=100, num_frames=25, k=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model (using weighted loss)
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device=device, class_weights=class_weights
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_peter_model.pth'))
    
    # Evaluate model
    predictions, targets, accuracy = evaluate_model(model, test_loader, device)
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(targets, predictions)
    
    # Print training summary
    print_training_summary(class_weights, accuracy)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 