import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset_fusion import FusionDataset
from model_fusion import FusionModel
import numpy as np

# --- CONFIG ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
MODEL_PATH = "best_fusion_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

def add_fog_noise(images, severity=0.8):
    """
    Injects Gaussian noise to simulate heavy sensor degradation (Fog/Grain).
    Severity 0.8 means the noise is almost as strong as the signal.
    """
    noise = torch.randn_like(images) * severity
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1) # Keep pixel values valid (0-1)

def main():
    print(f"--- Loading Fusion Model from {MODEL_PATH} ---")
    
    # 1. Setup Data
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Re-load dataset to get the validation split
    # Note: We must use the exact same random seed or split logic if we want exact comparison
    # But for robustness testing, a random split is fine.
    full_dataset = FusionDataset(ROOT_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_data = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    model = FusionModel().to(DEVICE)
    # Weights_only=False is needed for older pytorch/complex saves, safe here since we made the file
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)) 
    model.eval()

    print(f"\nEvaluating on {len(val_data)} samples...")
    print("="*65)
    print(f"{'SCENARIO':<25} | {'ACCURACY':<10} | {'RECALL (Avg)':<10}")
    print("="*65)

    # --- SCENARIO 1: CLEAR DAY (Baseline) ---
    evaluate_scenario(model, val_loader, mode="normal", title="1. Clear Conditions")

    # --- SCENARIO 2: HEAVY FOG (Noisy Image) ---
    evaluate_scenario(model, val_loader, mode="fog", title="2. Heavy Fog (Noise)")

    # --- SCENARIO 3: CAMERA FAILURE (Black Image) ---
    # This effectively tests the RADAR BRANCH only
    evaluate_scenario(model, val_loader, mode="blind", title="3. Camera Failure")
    
    print("="*65)

def evaluate_scenario(model, loader, mode, title):
    correct_preds = 0
    total_elements = 0
    
    # For per-class recall
    true_positives = torch.zeros(3).to(DEVICE)
    actual_positives = torch.zeros(3).to(DEVICE)

    with torch.no_grad():
        for images, radars, labels in loader:
            images = images.to(DEVICE)
            radars = radars.to(DEVICE)
            labels = labels.to(DEVICE)

            # --- MODIFY INPUTS BASED ON SCENARIO ---
            if mode == "fog":
                images = add_fog_noise(images, severity=0.8) # High noise
            elif mode == "blind":
                # Create a black tensor of the same shape
                images = torch.zeros_like(images) 

            # Forward Pass
            outputs = model(images, radars)
            
            # Metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Accuracy
            correct_preds += (preds == labels).sum().item()
            total_elements += labels.numel()

            # Recall Stats (TP / TP+FN)
            true_positives += (preds * labels).sum(dim=0)
            actual_positives += labels.sum(dim=0)

    # Calculate Final Stats
    acc = (correct_preds / total_elements) * 100
    
    # Average Recall across 3 classes
    recall_per_class = true_positives / (actual_positives + 1e-6)
    avg_recall = recall_per_class.mean().item() * 100

    print(f"{title:<25} | {acc:.2f}%     | {avg_recall:.2f}%")

if __name__ == '__main__':
    main()