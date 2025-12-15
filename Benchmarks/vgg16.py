import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import pandas as pd

# 1. Setup ResNet18 (Pre-trained on ImageNet)
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet has 1000 classes. We need to know which ones map to your dataset.
# Your Dataset: {0: Person, 2: Car, 3: Motorbike, 5: Bus, 7: Truck, 80: Cyclist}
# ImageNet indices (simplified examples):
imagenet_mapping = {
    'car': [817, 511, 479, 751, 656, 436], # sports car, convertible, jeep, etc.
    'person': [981, 982, 983], # Scuba diver, groom, ballplayer (ImageNet is bad at generic "people")
    'truck': [867, 864, 734], # Trailer truck, tow truck
    'motorbike': [670, 671]   # Motor scooter, moped
}

# 2. Transform (Resize to 224x224 for ResNet)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Path Setup
image_dir = './2019_04_09_bms1000/images_0/'
label_dir = './2019_04_09_bms1000/text_labels/'

total_frames = 0
correct_detections = 0

# 4. The Loop
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # Load Image
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        img_t = transform(img).unsqueeze(0)

        # Load Corresponding Label CSV (Same filename, but .csv)
        csv_name = filename.replace(".jpg", ".csv")
        csv_path = os.path.join(label_dir, csv_name)
        
        if os.path.exists(csv_path):
            # Read CSV: Format [uid, class, px, py, wid, len]
            # We only care about column 1 ('class')
            try:
                df = pd.read_csv(csv_path, header=None)
                gt_classes = df[1].unique() # Get all unique objects in this frame
            except:
                continue # Skip empty or malformed CSVs

            # Run Model
            with torch.no_grad():
                outputs = model(img_t)
            
            # Get Top 5 predictions from ResNet
            _, preds = torch.topk(outputs, 5)
            preds = preds[0].tolist()

            # CHECK: Did we find a Car?
            # Example: If CSV says "Car" (Class 2), did ResNet output any "Car" ID?
            
            # Let's test specifically for CARS (Class ID 2 in your data)
            if 2 in gt_classes: 
                total_frames += 1
                # Check if any of the top 5 predictions are in our ImageNet 'car' list
                if any(p in imagenet_mapping['car'] for p in preds):
                    correct_detections += 1

# 5. Final Metric
if total_frames > 0:
    print(f"ResNet18 Accuracy on 'Car' Class: {correct_detections/total_frames*100:.2f}%")
else:
    print("No frames with cars found or checked.")