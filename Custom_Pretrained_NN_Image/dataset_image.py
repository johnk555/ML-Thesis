import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        # --- CONFIGURATION ---
        # 1. Class Mapping: Map Dataset IDs to Neural Network Neurons (0, 1, 2)
        # We ignore Bus(5), Truck(7), Motorbike(3) for now to focus on the main classes.
        self.target_map = {
            0: 0,   # Person  -> Neuron 0
            80: 1,  # Cyclist -> Neuron 1
            2: 2    # Car     -> Neuron 2
        }
        
        # 2. Distance Limit (Meters)
        # We only train on objects closer than 25m to ensure high-quality features.
        self.max_distance = 25.0

        print("Scanning dataset for valid images and labels...")
        
        # Counters for sanity check
        counts = {0: 0, 1: 0, 2: 0} 
        
        for seq in os.listdir(root_dir):
            seq_path = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_path): continue
            
            img_dir = os.path.join(seq_path, 'images_0')
            lbl_dir = os.path.join(seq_path, 'text_labels')
            
            if not (os.path.exists(img_dir) and os.path.exists(lbl_dir)): continue
            
            for f in os.listdir(img_dir):
                if f.endswith('.jpg'):
                    csv_name = f.replace('.jpg', '.csv')
                    csv_path = os.path.join(lbl_dir, csv_name)
                    
                    if os.path.exists(csv_path):
                        # Pre-check: Does this image contain any VALID objects?
                        try:
                            # Read CSV: [uid, class, px, py, wid, len]
                            df = pd.read_csv(csv_path, header=None)
                            
                            # Filter: Keep rows where (Class is known) AND (Distance < 25m)
                            # Column 1 = Class, Column 3 = py (longitudinal distance)
                            valid_rows = df[ (df[1].isin(self.target_map.keys())) & (df[3].abs() < self.max_distance) ]
                            
                            if not valid_rows.empty:
                                self.samples.append((os.path.join(img_dir, f), csv_path))
                                
                                # Update counts for user info
                                for cls_id in valid_rows[1].unique():
                                    counts[self.target_map[cls_id]] += 1
                        except:
                            continue

        print(f"Dataset Ready. Found {len(self.samples)} valid frames.")
        print(f"Class Distribution in Training Set: Person: {counts[0]}, Cyclist: {counts[1]}, Car: {counts[2]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]
        
        # 1. Load Image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 2. Load Label (Multi-Hot Vector)
        # Shape [3]: [Person, Cyclist, Car]
        # Example: [1, 0, 1] means Person AND Car are in the image.
        label = torch.zeros(3, dtype=torch.float32)
        
        try:
            df = pd.read_csv(csv_path, header=None)
            
            # Get objects within range
            valid_rows = df[ (df[1].isin(self.target_map.keys())) & (df[3].abs() < self.max_distance) ]
            
            # Set the corresponding index to 1.0
            for cls_id in valid_rows[1].unique():
                idx = self.target_map[cls_id]
                label[idx] = 1.0
                
        except:
            pass 

        return image, label