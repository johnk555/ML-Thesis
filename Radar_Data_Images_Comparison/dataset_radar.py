import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import scipy.io as sio
import numpy as np

class RadarDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = [] 
        
        # 0=Person, 1=Cyclist, 2=Car
        self.target_map = {0: 0, 80: 1, 2: 2}

        print("Scanning dataset for Radar files...")
        
        for seq in os.listdir(root_dir):
            seq_path = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_path): continue
            
            radar_dir = os.path.join(seq_path, 'radar_raw_frame')
            lbl_dir = os.path.join(seq_path, 'text_labels')
            img_dir = os.path.join(seq_path, 'images_0')
            
            if not (os.path.exists(radar_dir) and os.path.exists(lbl_dir)): continue
            
            if os.path.exists(img_dir):
                file_list = os.listdir(img_dir)
            else:
                continue

            for f in file_list:
                if f.endswith('.jpg'):
                    img_basename = f.replace('.jpg', '') 
                    try:
                        frame_idx = int(img_basename)
                    except ValueError:
                        continue
                        
                    radar_name = f"{frame_idx:06d}.mat"
                    csv_name = f.replace('.jpg', '.csv')
                    
                    mat_path = os.path.join(radar_dir, radar_name)
                    csv_path = os.path.join(lbl_dir, csv_name)
                    
                    if os.path.exists(mat_path) and os.path.exists(csv_path):
                        # Verify class existence
                        try:
                            df = pd.read_csv(csv_path, header=None)
                            if any(cls_id in self.target_map for cls_id in df[1].unique()):
                                self.samples.append((mat_path, csv_path))
                        except:
                            continue

        print(f"Radar Dataset Ready. Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, csv_path = self.samples[idx]
        
        # --- 1. LOAD RADAR ---
        try:
            mat = sio.loadmat(mat_path)
            
            # --- FIX: Handle different naming conventions ---
            if 'adcData' in mat:
                raw_data = mat['adcData']
            elif 'adc_data' in mat:
                raw_data = mat['adc_data']
            elif 'Radar_Data' in mat:
                raw_data = mat['Radar_Data']
            else:
                raise KeyError("No valid radar data key found in .mat file")
            
            # Take antenna 0,0. Shape: (128, 255, 4, 2) -> (128, 255)
            adc_data = raw_data[:, :, 0, 0] 
            
            # FFT Processing
            range_doppler = np.fft.fft2(adc_data)
            rd_map = np.fft.fftshift(range_doppler)
            rd_map = np.log10(np.abs(rd_map) + 1)
            
            # Normalize 0-1
            denominator = rd_map.max() - rd_map.min()
            if denominator == 0: denominator = 1e-6
            rd_map = (rd_map - rd_map.min()) / denominator
            
            # Resize to square 128x128
            radar_tensor = torch.tensor(rd_map, dtype=torch.float32).unsqueeze(0)
            radar_tensor = torch.nn.functional.interpolate(radar_tensor.unsqueeze(0), size=(128, 128)).squeeze(0)
            
        except Exception as e:
            # print(f"Error loading {mat_path}: {e}") # Uncomment to debug specific files
            radar_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)

        # --- 2. LOAD LABEL ---
        label = torch.zeros(3, dtype=torch.float32)
        try:
            df = pd.read_csv(csv_path, header=None)
            for cls_id in df[1].unique():
                if cls_id in self.target_map:
                    label[self.target_map[cls_id]] = 1.0     
        except:
            pass 

        return radar_tensor, label