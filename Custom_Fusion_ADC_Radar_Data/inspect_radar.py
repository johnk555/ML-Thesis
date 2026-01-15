import torch
from torch.utils.data import DataLoader
from dataset_radar import RadarDataset
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = r'C:/Users/karel/Automotive' 

def main():
    print("--- Inspecting Radar Data ---")
    
    # 1. Load Dataset
    dataset = RadarDataset(ROOT_DIR)
    
    # 2. Get one sample manually (bypass DataLoader for a second)
    print(f"\nChecking sample index 0...")
    radar, label = dataset[0]
    
    print(f"Tensor Shape: {radar.shape}")
    print(f"Max Value: {radar.max().item()}")
    print(f"Min Value: {radar.min().item()}")
    print(f"Mean Value: {radar.mean().item()}")
    
    # 3. Check for "Dead" Data
    if radar.max().item() == 0 and radar.min().item() == 0:
        print("\n❌ CRITICAL ERROR: The radar tensor is all ZEROS.")
        print("This means the loading logic inside __getitem__ is failing and triggering the 'except' block.")
    else:
        print("\n✅ Data looks alive (not all zeros).")
        
        # Optional: Visualize
        plt.imshow(radar.squeeze(), cmap='jet')
        plt.title(f"Radar Sample (Label: {label})")
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    main()