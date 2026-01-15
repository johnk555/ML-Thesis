import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset_fusion import FusionDataset
from model_fusion import FusionModel
import time

# --- UPDATE PATH ---
ROOT_DIR = r'C:/Users/karel/Automotive' 

# Config
BATCH_SIZE = 16 # Lower batch size because we are loading 2 sets of data
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting FUSION Training on {DEVICE} ---")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Load Dataset
    full_dataset = FusionDataset(ROOT_DIR, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training Samples: {train_size} | Validation Samples: {val_size}")
    
    # 2. Loaders
    # num_workers=0 is safer for Windows + FFT processing
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    model = FusionModel().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() 
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        # --- TRAINING ---
        for images, radars, labels in train_loader:
            images = images.to(DEVICE)
            radars = radars.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass with TWO inputs
            outputs = model(images, radars)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_elements = 0
        
        with torch.no_grad():
            for images, radars, labels in val_loader:
                images = images.to(DEVICE)
                radars = radars.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images, radars)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct_preds += (preds == labels).sum().item()
                total_elements += labels.numel()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_preds / total_elements
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_fusion_model.pth")
            print("  --> Model Saved")

    print("\nFusion Training Complete.")

if __name__ == '__main__':
    main()