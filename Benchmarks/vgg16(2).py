import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import pandas as pd

# ==========================================
# 1. SETUP: Define the Models & Mappings
# ==========================================

# Select your model here: 'resnet18', 'vgg16', or 'efficientnet'
MODEL_NAME = 'resnet18' 

# Load the Pre-trained Model
print(f"Loading {MODEL_NAME}...")
if MODEL_NAME == 'resnet18':
    model = models.resnet18(pretrained=True)
elif MODEL_NAME == 'vgg16':
    model = models.vgg16(pretrained=True)
elif MODEL_NAME == 'efficientnet':
    model = models.efficientnet_b0(pretrained=True)

model.eval() # Set to evaluation mode (no training)

# Define Image Transform (Resize to model input size)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MAPPING: Your CSV Class IDs vs. ImageNet Class IDs
# Your Dataset: {0: Person, 2: Car, 3: Motorbike, 5: Bus, 7: Truck, 80: Cyclist}
dataset_class_id = 2  # Let's test for CARS first
target_name = "Car"

# ImageNet has 1000 classes. These are the IDs that represent "Cars"
imagenet_car_ids = [
    817, 511, 479, 751, 656, 436, 705, 751, 817, 468, 609, 627, 
    654, 661, 717, 864, 656, 436 # Sports car, jeep, taxi, minivan, etc.
]

# ==========================================
# 2. EXECUTION: The Loop
# ==========================================
# UPDATE THESE PATHS to your actual folders
image_dir = 'C:/Users/karel/Automotive/2019_04_09_bms1000/images_0'
label_dir = 'C:/Users/karel/Automotive/2019_04_09_bms1000/text_labels'

total_car_images = 0
correctly_detected = 0

print(f"Starting evaluation for class: {target_name}...")

# Iterate through every JPG in your folder
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        
        # 1. READ THE CSV LABEL
        csv_name = filename.replace(".jpg", ".csv")
        csv_path = os.path.join(label_dir, csv_name)
        
        has_target_object = False
        
        if os.path.exists(csv_path):
            try:
                # Read CSV. Columns are: [uid, class, px, py, wid, len]
                # We check column index 1 ('class')
                df = pd.read_csv(csv_path, header=None)
                objects_in_frame = df[1].unique()
                
                if dataset_class_id in objects_in_frame:
                    has_target_object = True
            except Exception:
                continue # Skip corrupt/empty CSVs
        
        # If the CSV says there is NO car, we skip this image (we only want Recall for now)
        if not has_target_object:
            continue
            
        total_car_images += 1
        
        # 2. RUN THE MODEL
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Get Top 5 Predictions
            _, top5_indices = torch.topk(outputs, 5)
            top5_list = top5_indices[0].tolist()
            
            # 3. CHECK MATCH
            # Did the model predict ANY class that looks like a "Car"?
            match_found = any(idx in imagenet_car_ids for idx in top5_list)
            
            if match_found:
                correctly_detected += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# ==========================================
# 3. RESULTS
# ==========================================
if total_car_images > 0:
    accuracy = (correctly_detected / total_car_images) * 100
    print("-" * 30)
    print(f"RESULTS FOR {MODEL_NAME.upper()}")
    print(f"Total Images containing {target_name}: {total_car_images}")
    print(f"Correctly Detected: {correctly_detected}")
    print(f"Recall Accuracy: {accuracy:.2f}%")
    print("-" * 30)
else:
    print("No images with the target class were found in the folder.")