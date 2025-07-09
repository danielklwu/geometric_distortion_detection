# Geometric Distortion Detection and Classification System
# Description: A complete system for detecting and classifying geometric distortions in images

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
import json
from typing import Tuple, List, Dict, Optional
import argparse
from pathlib import Path

# Configuration
class Config:
    # Dataset configuration
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Distortion types
    DISTORTION_TYPES = {
        'none': 0,
        'rotation': 1,
        'scaling': 2,
        'translation': 3,
        'perspective': 4,
        'shearing': 5
    }
    
    # Model configuration
    MODEL_PATH = 'models/distortion_classifier.pth'
    DATASET_PATH = 'dataset'
    
    # Visualization
    VISUALIZE_SAMPLES = True
    TOP_N_PREDICTIONS = 3

class GeometricDistortionGenerator:
    """Generate synthetic dataset with various geometric distortions."""
    
    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each distortion type
        for distortion_type in Config.DISTORTION_TYPES.keys():
            (self.output_dir / distortion_type).mkdir(exist_ok=True)
    
    def apply_rotation(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """Apply rotation distortion."""
        if angle is None:
            angle = random.uniform(-45, 45)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def apply_scaling(self, image: np.ndarray, scale_factor: float = None) -> np.ndarray:
        """Apply scaling distortion."""
        if scale_factor is None:
            scale_factor = random.uniform(0.5, 2.0)
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Scale image
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale_factor > 1.0:
            # Crop from center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            scaled = scaled[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad to original size
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_y, h - new_h - pad_y, 
                                       pad_x, w - new_w - pad_x, 
                                       cv2.BORDER_REFLECT)
        
        return scaled
    
    def apply_translation(self, image: np.ndarray, tx: int = None, ty: int = None) -> np.ndarray:
        """Apply translation distortion."""
        h, w = image.shape[:2]
        
        if tx is None:
            tx = random.randint(-w//4, w//4)
        if ty is None:
            ty = random.randint(-h//4, h//4)
        
        # Translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        translated = cv2.warpAffine(image, translation_matrix, (w, h), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REFLECT)
        return translated
    
    def apply_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective distortion."""
        h, w = image.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points with random perspective distortion
        max_offset = min(w, h) // 8
        dst_points = np.float32([
            [random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
            [random.randint(0, max_offset), h - random.randint(0, max_offset)]
        ])
        
        # Get perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        perspective = cv2.warpPerspective(image, perspective_matrix, (w, h), 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_REFLECT)
        return perspective
    
    def apply_shearing(self, image: np.ndarray, shear_factor: float = None) -> np.ndarray:
        """Apply shearing distortion."""
        if shear_factor is None:
            shear_factor = random.uniform(-0.3, 0.3)
        
        h, w = image.shape[:2]
        
        # Shearing matrix
        shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        
        # Apply shearing
        sheared = cv2.warpAffine(image, shear_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)
        return sheared
    
    def generate_sample_images(self, num_samples: int = 100):
        """Generate sample images using basic shapes and patterns."""
        samples = []
        
        for i in range(num_samples):
            # Create a blank image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Add random geometric shapes
            shape_type = random.choice(['rectangle', 'circle', 'triangle', 'lines'])
            
            if shape_type == 'rectangle':
                pt1 = (random.randint(10, 100), random.randint(10, 100))
                pt2 = (random.randint(120, 200), random.randint(120, 200))
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                cv2.rectangle(img, pt1, pt2, color, -1)
                
            elif shape_type == 'circle':
                center = (random.randint(50, 174), random.randint(50, 174))
                radius = random.randint(20, 50)
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                cv2.circle(img, center, radius, color, -1)
                
            elif shape_type == 'triangle':
                pts = np.array([
                    [random.randint(10, 214), random.randint(10, 214)],
                    [random.randint(10, 214), random.randint(10, 214)],
                    [random.randint(10, 214), random.randint(10, 214)]
                ], np.int32)
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                cv2.fillPoly(img, [pts], color)
                
            elif shape_type == 'lines':
                for _ in range(random.randint(3, 8)):
                    pt1 = (random.randint(0, 224), random.randint(0, 224))
                    pt2 = (random.randint(0, 224), random.randint(0, 224))
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    cv2.line(img, pt1, pt2, color, random.randint(2, 5))
            
            # Add some noise for variety
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            samples.append(img)
        
        return samples
    
    def create_dataset(self, num_samples_per_class: int = 500):
        """Create a complete dataset with all distortion types."""
        print("Generating base images...")
        base_images = self.generate_sample_images(num_samples_per_class)
        
        distortion_functions = {
            'none': lambda x: x,
            'rotation': self.apply_rotation,
            'scaling': self.apply_scaling,
            'translation': self.apply_translation,
            'perspective': self.apply_perspective,
            'shearing': self.apply_shearing
        }
        
        dataset_info = []
        
        for distortion_type, func in distortion_functions.items():
            print(f"Generating {distortion_type} samples...")
            
            for i, base_image in enumerate(base_images):
                # Apply distortion
                distorted_image = func(base_image.copy())
                
                # Save image
                filename = f"{distortion_type}_{i:04d}.png"
                filepath = self.output_dir / distortion_type / filename
                cv2.imwrite(str(filepath), distorted_image)
                
                # Record metadata
                dataset_info.append({
                    'filename': str(filepath),
                    'distortion_type': distortion_type,
                    'class_id': Config.DISTORTION_TYPES[distortion_type]
                })
        
        # Save dataset info
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset created with {len(dataset_info)} samples")
        return dataset_info

class DistortionDataset(Dataset):
    """PyTorch Dataset for distortion classification."""
    
    def __init__(self, dataset_info: List[Dict], transform=None):
        self.dataset_info = dataset_info
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset_info)
    
    def __getitem__(self, idx):
        item = self.dataset_info[idx]
        
        # Load image
        image = cv2.imread(item['filename'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = item['class_id']
        
        return image, label

class DistortionClassifier(nn.Module):
    """Custom classifier based on pre-trained ResNet50."""
    
    def __init__(self, num_classes: int = 6):
        super(DistortionClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone layers (optional - can be unfrozen for fine-tuning)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Unfreeze the final layers for training
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

class DistortionDetector:
    """Main class for training and inference."""
    
    def __init__(self, model_path: str = Config.MODEL_PATH):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = DistortionClassifier(len(Config.DISTORTION_TYPES))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, dataset_info: List[Dict]):
        """Prepare training and validation datasets."""
        # Split dataset
        random.shuffle(dataset_info)
        split_idx = int(0.8 * len(dataset_info))
        
        train_info = dataset_info[:split_idx]
        val_info = dataset_info[split_idx:]
        
        # Create datasets
        train_dataset = DistortionDataset(train_info, self.train_transform)
        val_dataset = DistortionDataset(val_info, self.val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                               shuffle=False, num_workers=4)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(Config.NUM_EPOCHS):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}, '
                          f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.6f}')
            
            # Validation phase
            val_acc = self.validate(val_loader)
            
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}: '
                  f'Train Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_path)
                print(f'New best model saved with validation accuracy: {best_val_acc:.4f}')
            
            scheduler.step()
        
        return train_losses, val_accuracies
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    def load_model(self):
        """Load trained model."""
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def predict(self, image_path: str, top_n: int = Config.TOP_N_PREDICTIONS):
        """Predict distortion type for a single image."""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top N predictions
            top_probs, top_indices = torch.topk(probabilities, top_n)
            
            results = []
            for i in range(top_n):
                class_id = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                
                # Find class name
                class_name = None
                for name, id in Config.DISTORTION_TYPES.items():
                    if id == class_id:
                        class_name = name
                        break
                
                results.append({
                    'distortion_type': class_name,
                    'confidence': confidence
                })
        
        return results

def visualize_samples(dataset_info: List[Dict], num_samples: int = 12):
    """Visualize sample images from each distortion type."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('Sample Images from Each Distortion Type', fontsize=16)
    
    # Group by distortion type
    samples_by_type = {}
    for item in dataset_info:
        distortion_type = item['distortion_type']
        if distortion_type not in samples_by_type:
            samples_by_type[distortion_type] = []
        samples_by_type[distortion_type].append(item)
    
    # Plot samples
    for i, (distortion_type, samples) in enumerate(samples_by_type.items()):
        if i >= 6:  # Only show first 6 types
            break
            
        # Select two random samples
        random_samples = random.sample(samples, min(2, len(samples)))
        
        for j, sample in enumerate(random_samples):
            image = cv2.imread(sample['filename'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[j, i].imshow(image)
            axes[j, i].set_title(f'{distortion_type.title()}')
            axes[j, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='Geometric Distortion Detection System')
    parser.add_argument('--mode', choices=['generate', 'train', 'predict'], 
                       default='train', help='Mode to run')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--num_samples', type=int, default=500, 
                       help='Number of samples per distortion type')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Generate dataset
        generator = GeometricDistortionGenerator()
        dataset_info = generator.create_dataset(args.num_samples)
        
        if Config.VISUALIZE_SAMPLES:
            visualize_samples(dataset_info)
    
    elif args.mode == 'train':
        # Load or generate dataset
        dataset_info_path = Path(Config.DATASET_PATH) / 'dataset_info.json'
        
        if not dataset_info_path.exists():
            print("Dataset not found. Generating dataset...")
            generator = GeometricDistortionGenerator()
            dataset_info = generator.create_dataset(args.num_samples)
        else:
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
        
        # Train model
        detector = DistortionDetector()
        train_loader, val_loader = detector.prepare_data(dataset_info)
        
        print("Starting training...")
        train_losses, val_accuracies = detector.train(train_loader, val_loader)
        
        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()
    
    elif args.mode == 'predict':
        if not args.image_path:
            print("Please provide --image_path for prediction")
            return
        
        # Load model and predict
        detector = DistortionDetector()
        detector.load_model()
        
        results = detector.predict(args.image_path)
        
        print(f"\nPrediction results for {args.image_path}:")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['distortion_type'].title()}: "
                  f"{result['confidence']:.4f}")
        
        # Visualize result
        image = cv2.imread(args.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Predicted: {results[0]['distortion_type'].title()} "
                 f"(Confidence: {results[0]['confidence']:.4f})")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()