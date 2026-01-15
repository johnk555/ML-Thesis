import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import scipy.io as sio
import numpy as np
from PIL import Image

class FusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        # 0=Person, 1=Cyclist, 2=Car
        self.target_map = {0: 0, 80: 1, 2: 2}

        print("Scanning dataset for paired Image+Radar files...")
        
        for seq in os.listdir(root_dir):
            seq_path = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_path): continue
            
            img_dir = os.path.join(seq_path, 'images_0')
            radar_dir = os.path.join(seq_path, 'radar_raw_frame')
            lbl_dir = os.path.join(seq_path, 'text_labels')
            
            if not (os.path.exists(img_dir) and os.path.exists(radar_dir) and os.path.exists(lbl_dir)): continue
            
            for f in os.listdir(img_dir):
                if f.endswith('.jpg'):
                    # --- FILENAME FIX START ---
                    # Image: "0000000000.jpg" (10 digits)
                    img_basename = f.replace('.jpg', '') 
                    
                    try:
                        frame_idx = int(img_basename) # Becomes integer 0
                    except ValueError:
                        continue # Skip non-numbered files
                        
                    # Radar: "000000.mat" (6 digits)
                    radar_name = f"{frame_idx:06d}.mat"
                    
                    # Label: "0000000000.csv" (Matches image)
                    csv_name = f.replace('.jpg', '.csv')
                    # --- FILENAME FIX END ---

                    csv_path = os.path.join(lbl_dir, csv_name)
                    mat_path = os.path.join(radar_dir, radar_name)
                    
                    if os.path.exists(csv_path) and os.path.exists(mat_path):
                        # Pre-check: Does this frame contain Person, Car, or Cyclist?
                        try:
                            df = pd.read_csv(csv_path, header=None)
                            if any(cls_id in self.target_map for cls_id in df[1].unique()):
                                self.samples.append((os.path.join(img_dir, f), mat_path, csv_path))
                        except:
                            continue

        print(f"Fusion Dataset Ready. Found {len(self.samples)} valid pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mat_path, csv_path = self.samples[idx]
        
        # --- 1. IMAGE ---
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # --- 2. RADAR ---
        try:
            mat = sio.loadmat(mat_path)
            key = 'adc_data' if 'adc_data' in mat else 'Radar_Data'
            raw_data = mat[key] 
            
            # Take antenna 0,0
            adc_data = raw_data[:, :, 0, 0] 
            
            # FFT Processing
            range_doppler = np.fft.fft2(adc_data)
            rd_map = np.fft.fftshift(range_doppler)
            rd_map = np.log10(np.abs(rd_map) + 1)
            
            # Normalize 0-1
            rd_map = (rd_map - rd_map.min()) / (rd_map.max() - rd_map.min())
            
            # Tensor (1 Channel, Resize to 128x128)
            radar_tensor = torch.tensor(rd_map, dtype=torch.float32).unsqueeze(0)
            radar_tensor = torch.nn.functional.interpolate(radar_tensor.unsqueeze(0), size=(128, 128)).squeeze(0)
            
        except Exception:
            # Fallback (Empty radar map)
            radar_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)

        # --- 3. LABEL ---
        label = torch.zeros(3, dtype=torch.float32)
        try:
            df = pd.read_csv(csv_path, header=None)
            for cls_id in df[1].unique():
                if cls_id in self.target_map:
                    label[self.target_map[cls_id]] = 1.0     
        except:
            pass 

        return image, radar_tensor, label