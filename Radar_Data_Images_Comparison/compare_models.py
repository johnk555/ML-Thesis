import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset_fusion import FusionDataset
from model_image import CustomImageModel
from model_radar import RadarModel
import pandas as pd

# --- CONFIG ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
IMAGE_MODEL_PATH = "custom_scratch_model.pth" # Ensure this file is in the folder!
RADAR_MODEL_PATH = "best_radar_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("--- Loading Independent Models ---")
    
    # 1. Setup Data
    # We use FusionDataset because it gives us BOTH Image and Radar for the same frame
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = FusionDataset(ROOT_DIR, transform=transform)
    
    # We only want to look at the validation set (unseen data)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Batch size 1 makes it easy to print line-by-line
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    
    # 2. Load Image Model
    print(f"Loading Image Model from {IMAGE_MODEL_PATH}...")
    model_img = CustomImageModel().to(DEVICE)
    # We use strict=False because your saved model might have a 'classifier' layer 
    # that matches differently depending on how you saved it. usually strict=True is best.
    try:
        model_img.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        # Fallback if keys don't match perfectly (common when moving files)
        print("Warning: Strict loading failed, trying flexible loading...")
        model_img.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model_img.eval()

    # 3. Load Radar Model
    print(f"Loading Radar Model from {RADAR_MODEL_PATH}...")
    model_rad = RadarModel().to(DEVICE)
    model_rad.load_state_dict(torch.load(RADAR_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model_rad.eval()

    print("\n--- STARTING COMPARISON (Press Ctrl+C to stop) ---")
    print(f"{'TYPE':<10} | {'IMAGE PRED':<20} | {'RADAR PRED':<20} | {'AGREEMENT?'}")
    print("-" * 75)

    classes = ["Person", "Cyclist", "Car"]

    with torch.no_grad():
        for i, (image, radar, label) in enumerate(val_loader):
            image = image.to(DEVICE)
            radar = radar.to(DEVICE)
            label = label.to(DEVICE)

            # --- Get Predictions ---
            # Image
            out_img = model_img(image)
            prob_img = torch.sigmoid(out_img)[0] # Get prob of first sample
            
            # Radar
            out_rad = model_rad(radar)
            prob_rad = torch.sigmoid(out_rad)[0]

            # --- Logic to find the "Main" prediction ---
            # We look for the class with the highest probability
            img_cls_idx = torch.argmax(prob_img).item()
            rad_cls_idx = torch.argmax(prob_rad).item()
            
            # Ground Truth (Label is one-hot, e.g. [1, 0, 0])
            gt_idx = torch.argmax(label[0]).item()
            gt_name = classes[gt_idx]

            img_name = classes[img_cls_idx]
            rad_name = classes[rad_cls_idx]
            
            img_conf = prob_img[img_cls_idx].item() * 100
            rad_conf = prob_rad[rad_cls_idx].item() * 100

            # Only print valid objects (skip if label is all zeros/background)
            if label.sum().item() == 0: continue

            # Formatting Output
            #match = "✅ YES" if img_cls_idx == rad_cls_idx else "❌ DISAGREE"

            if img_cls_idx == rad_cls_idx:
                if img_cls_idx == gt_idx:
                    match = "✅ Both Correct"
                else:
                    match = "⚠️ Shared Error" # This is the dangerous category
            elif img_cls_idx == gt_idx:
                match = "📷 Img Wins"
            elif rad_cls_idx == gt_idx:
                match = "📡 Rad Wins"
            else:
                match = "❌ Total Failure"
            
            print(f"GT: {gt_name:<6} | Img: {img_name} ({img_conf:.0f}%)    | Rad: {rad_name} ({rad_conf:.0f}%)    | {match}")

            # Stop after 20 samples so we don't flood the screen
            if i > 20: 
                break

if __name__ == '__main__':
    main()