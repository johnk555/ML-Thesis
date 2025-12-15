import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Use 'r' before the string to handle Windows backslashes automatically
ROOT_DIR = r'C:/Users/karel/Automotive' 

# Choose Model: 'resnet18', 'vgg16', 'efficientnet'
MODEL_NAME = 'efficientnet' 

# Target Class to Detect (Default: Car)
# Dataset IDs: {0: Person, 2: Car, 3: Motorbike, 5: Bus, 7: Truck, 80: Cyclist}
DATASET_CLASS_ID = 0
TARGET_NAME = "Person"

# --- CARS ---
IMAGENET_CAR_IDS = [
        436, # beach wagon, station wagon
        468, # cab, hack, taxi
        511, # convertible
        609, # jeep, landrover
        627, # limousine
        656, # minivan
        717, # pickup, pickup truck
        751, # racer, race car
        817, # sports car, sport car
        864, # tow truck (often looks like a car/truck hybrid)
        479, # car wheel (sometimes the model only sees the wheel)
        581, # grille, radiator grille (front of car)
]

# --- TRUCKS ---
IMAGENET_TRUCK_IDS = [
        867, # trailer truck, tractor trailer, trucking rig
        569, # garbage truck, dustcart
        555, # fire engine, fire truck
        675, # moving van
        734, # police van
        864, # tow truck (can be here too)
        654, # minibus (often looks like a delivery truck)
]

# Note: ImageNet is weak on "heavy" motorcycles (Harleys/Sport bikes) 
# but these are the closest valid IDs.
# --- BUSES ---
IMAGENET_BUS_IDS = [
        779, # school bus
        829, # streetcar, tram (often mistaken for bus)
        874, # trolleybus, trolley coach
        654, # minibus
]

# --- MOTORBIKES ---
IMAGENET_MOTORBIKE_IDS = [
        670, # motor scooter, scooter
        665, # moped
]

# --- CYCLISTS ---
# ImageNet detects the BICYCLE, not the human rider.
IMAGENET_CYCLISTS_IDS = [
        444, # bicycle-built-for-two, tandem bicycle
        671, # mountain bike, all-terrain bike
        870, # tricycle
        425, # barn, sheds (sometimes bikes are near structures, but risky to include)
]

# --- PERSON ---
# WARNING: ImageNet has NO generic "Pedestrian" class. 
# It only has specific "roles" (Scuba diver, Groom, Baseball player).
# These are the only ones that might resemble a pedestrian on a street.
IMAGENET_PERSON_IDS = [
        981, # ballplayer, baseball player
        982, # groom, bridegroom
        983, # scuba diver (rarely useful)
]

# What the algorithm will search for
SEARCH_FOR = IMAGENET_PERSON_IDS

# ==========================================
# 2. MODEL SETUP
# ==========================================

print(f"--- Loading {MODEL_NAME.upper()} (Pre-trained) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

if MODEL_NAME == 'resnet18':
    model = models.resnet18(pretrained=True)
elif MODEL_NAME == 'vgg16':
    model = models.vgg16(pretrained=True)
elif MODEL_NAME == 'efficientnet':
    model = models.efficientnet_b0(pretrained=True)

model = model.to(device)
model.eval()

# Standard ImageNet Transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. THE CRAWLER & EVALUATOR
# ==========================================
global_total_targets = 0
global_correct_detections = 0

print(f"\nScanning directory: {ROOT_DIR}\n")

# Walk through all folders
for root, dirs, files in os.walk(ROOT_DIR):
    
    # We look for folders that contain an 'images_0' subfolder
    if 'images_0' in dirs:
        sequence_name = os.path.basename(root)
        image_folder = os.path.join(root, 'images_0')
        label_folder = os.path.join(root, 'text_labels') # Assuming text_labels is sibling to images_0
        
        # Check if label folder exists
        if not os.path.exists(label_folder):
            print(f"Skipping {sequence_name}: 'text_labels' folder missing.")
            continue

        # Initialize Sequence Stats
        seq_targets = 0
        seq_correct = 0
        
        print(f"Processing Sequence: {sequence_name} ...")
        
        # Iterate over images in this specific sequence
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
        for img_file in image_files:
            # 1. Resolve Paths
            img_path = os.path.join(image_folder, img_file)
            csv_file = img_file.replace('.jpg', '.csv')
            csv_path = os.path.join(label_folder, csv_file)
            
            # 2. Check Ground Truth (CSV)
            has_target = False
            if os.path.exists(csv_path):
                try:
                    # Read CSV (Column 1 is Class ID)
                    df = pd.read_csv(csv_path, header=None)
                    if DATASET_CLASS_ID in df[1].values:
                        has_target = True
                except:
                    continue # Skip bad CSVs
            
            if not has_target:
                continue # Skip images that don't have our target object
                
            seq_targets += 1
            
            # 3. Run Model Prediction
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                
                # Check Top 5 Predictions
                _, top5 = torch.topk(outputs, 5)
                top5 = top5[0].tolist()
                
                # 4. Match?
                if any(idx in SEARCH_FOR for idx in top5):
                    seq_correct += 1
                    
            except Exception as e:
                print(f"  Error on {img_file}: {e}")

        # --- Report for this Sequence ---
        if seq_targets > 0:
            acc = (seq_correct / seq_targets) * 100
            print(f"  > Found {seq_targets} {TARGET_NAME}s. Detected: {seq_correct}. Accuracy: {acc:.2f}%")
        else:
            print(f"  > No {TARGET_NAME}s found in this sequence.")
            
        # Update Global Stats
        global_total_targets += seq_targets
        global_correct_detections += seq_correct

# ==========================================
# 4. FINAL REPORT
# ==========================================
print("\n" + "="*40)
print(f"FINAL REPORT: {MODEL_NAME.upper()}")
print("="*40)
print(f"Total Images Checked (containing {TARGET_NAME}): {global_total_targets}")
print(f"Total Correctly Detected: {global_correct_detections}")

if global_total_targets > 0:
    global_acc = (global_correct_detections / global_total_targets) * 100
    print(f"GLOBAL RECALL ACCURACY: {global_acc:.2f}%")
else:
    print("No target objects found in the entire dataset path.")
print("="*40)