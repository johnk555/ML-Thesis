import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset_radar import RadarDataset
from model_radar import RadarModel
import time

# --- CONFIG ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
BATCH_SIZE = 32 
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting RADAR-ONLY Training on {DEVICE} ---")
    
    # 1. Load Dataset
    # No transforms needed (FFT and normalization are handled in dataset class)
    full_dataset = RadarDataset(ROOT_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training Samples: {train_size} | Validation Samples: {val_size}")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    model = RadarModel().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() 
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        # --- TRAINING ---
        for radars, labels in train_loader:
            radars = radars.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(radars) # Forward pass only with radar
            
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
            for radars, labels in val_loader:
                radars = radars.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(radars)
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
            torch.save(model.state_dict(), "best_radar_model.pth")
            print("  --> Model Saved")

    print("\nRadar-Only Training Complete.")

if __name__ == '__main__':
    main()