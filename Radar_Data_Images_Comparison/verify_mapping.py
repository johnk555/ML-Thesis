import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
from torchvision import transforms

# --- CONFIGURATION ---
DATA_PATH = r'C:/Users/karel/Automotive'
IMAGE_MODEL_PATH = "custom_scratch_model.pth"
RADAR_MODEL_PATH = "best_radar_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# 1. DEFINE CLASSES AS THE MODEL SEES THEM
# Based on our analysis: 0=Person, 1=Car, 2=Cyclist
MODEL_CLASSES = ['Person', 'Car', 'Cyclist']

def evaluate_with_remapping():
    print(f"--- Loading Models on {DEVICE} ---")
    
    # Load Models
    from model_image import CustomImageModel 
    from model_radar import RadarModel
    
    try:
        model_img = CustomImageModel().to(DEVICE)
        model_rad = RadarModel().to(DEVICE)
    except:
        model_img = CustomImageModel(num_classes=3).to(DEVICE)
        model_rad = RadarModel(num_classes=3).to(DEVICE)

    model_img.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model_rad.load_state_dict(torch.load(RADAR_MODEL_PATH, map_location=DEVICE, weights_only=False))
    
    model_img.eval()
    model_rad.eval()

    # 2. Load Dataset
    print(f"--- Scanning Dataset: {DATA_PATH} ---")
    from dataset_fusion import FusionDataset 
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = FusionDataset(DATA_PATH, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data Loaded. Evaluating {len(test_dataset)} samples...")

    # 3. Storage
    all_targets = []
    all_preds_img = []
    all_preds_rad = []
    failed_samples = []

    print("--- Starting Evaluation (Applying Class Remapping) ---")
    
    # 4. Evaluation Loop
    with torch.no_grad():
        for batch_idx, (images, radars, labels) in enumerate(test_loader):
            images, radars, labels = images.to(DEVICE), radars.to(DEVICE), labels.to(DEVICE)

            # Predict
            out_img = model_img(images)
            out_rad = model_rad(radars)
            _, pred_img = torch.max(out_img, 1)
            _, pred_rad = torch.max(out_rad, 1)

            # --- CRITICAL: FIX LABELS ---
            # Dataset gives: 0=Car, 1=Cyclist, 2=Person
            # Model expects: 1=Car, 2=Cyclist, 0=Person
            
            # We must map DATASET indices to MODEL indices
            # 0 (Car) -> 1
            # 1 (Cyclist) -> 2
            # 2 (Person) -> 0
            
            # Handle One-Hot if present
            if labels.ndim > 1 and labels.shape[1] > 1:
                labels_raw = torch.argmax(labels, dim=1)
            else:
                labels_raw = labels

            # Apply Mapping
            targets_mapped = labels_raw.clone()
            targets_mapped[labels_raw == 0] = 1 # Car -> 1
            targets_mapped[labels_raw == 1] = 2 # Cyclist -> 2
            targets_mapped[labels_raw == 2] = 0 # Person -> 0
            
            # -----------------------------

            # Convert to numpy
            batch_targets = targets_mapped.cpu().numpy().flatten().astype(int)
            batch_preds_img = pred_img.cpu().numpy().flatten().astype(int)
            batch_preds_rad = pred_rad.cpu().numpy().flatten().astype(int)

            all_targets.extend(batch_targets)
            all_preds_img.extend(batch_preds_img)
            all_preds_rad.extend(batch_preds_rad)

            # Log Failures
            start_idx = batch_idx * BATCH_SIZE
            for i in range(len(batch_targets)):
                if batch_preds_img[i] != batch_targets[i] or batch_preds_rad[i] != batch_targets[i]:
                    try:
                        fname = test_dataset.image_paths[start_idx + i]
                    except:
                        fname = f"File_{start_idx + i}"
                    
                    failed_samples.append({
                        "file": fname,
                        "truth": MODEL_CLASSES[batch_targets[i]],
                        "img": MODEL_CLASSES[batch_preds_img[i]],
                        "rad": MODEL_CLASSES[batch_preds_rad[i]]
                    })

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}...", end='\r')

    # 5. Report
    print("\n\n" + "="*40)
    print("FINAL PERFORMANCE REPORT")
    print("="*40)
    print(f"Classes: {MODEL_CLASSES}\n")
    
    print("--- IMAGE MODEL ---")
    print(classification_report(all_targets, all_preds_img, target_names=MODEL_CLASSES))
    print(confusion_matrix(all_targets, all_preds_img))

    print("\n--- RADAR MODEL ---")
    print(classification_report(all_targets, all_preds_rad, target_names=MODEL_CLASSES))
    print(confusion_matrix(all_targets, all_preds_rad))
    
    # Save Failures
    with open("failures_final.txt", "w") as f:
        f.write("Filename | Truth | Image Pred | Radar Pred\n")
        for fail in failed_samples:
            f.write(f"{fail['file']} | {fail['truth']} | {fail['img']} | {fail['rad']}\n")
            
    print(f"\n✅ Report Saved. Failures logged to 'failures_final.txt'")

if __name__ == "__main__":
    evaluate_with_remapping()