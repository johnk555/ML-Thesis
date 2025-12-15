import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset_image import ImageDataset
from model_image import CustomImageModel
import time

# --- CONFIGURATION ---
# UPDATE THIS PATH TO YOUR DATASET ROOT
ROOT_DIR = r'C:/Users/karel/Automotive' 

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10 # We can stop early if accuracy is good
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. Prepare Data Transforms
    # Standard normalization for ResNet
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load Dataset
    full_dataset = ImageDataset(ROOT_DIR, transform=transform)
    
    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training Samples: {train_size} | Validation Samples: {val_size}")
    
    # Drop_last=True helps with Batch Normalization stability
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Setup Model
    model = CustomImageModel().to(DEVICE)
    
    # 4. Optimizer & Loss
    # We only optimize parameters that require_grad (the classifier head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # BCEWithLogitsLoss is MANDATORY for Multi-Label classification
    # (Because one image can have BOTH a Car and a Person)
    criterion = nn.BCEWithLogitsLoss() 
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        # --- TRAINING PHASE ---
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_elements = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Calculate Accuracy (Threshold = 0.5)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                correct_preds += (preds == labels).sum().item()
                total_elements += labels.numel() # Total number of individual class predictions
        
        # Statistics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_preds / total_elements
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_image_model.pth")
            print("  --> Model Saved (Improved Validation Loss)")

    print("\nTraining Complete. Best model saved as 'best_image_model.pth'")

if __name__ == '__main__':
    main()