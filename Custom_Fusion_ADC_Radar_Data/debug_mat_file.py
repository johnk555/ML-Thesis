import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
ROOT_DIR = r'C:/Users/karel/Automotive' 

def main():
    print(f"--- Searching for a valid .mat file in {ROOT_DIR} ---")
    
    found_file = None
    
    # 1. Hunt for the first .mat file
    for seq in os.listdir(ROOT_DIR):
        radar_dir = os.path.join(ROOT_DIR, seq, 'radar_raw_frame')
        if not os.path.exists(radar_dir): continue
        
        for f in os.listdir(radar_dir):
            if f.endswith('.mat'):
                found_file = os.path.join(radar_dir, f)
                break
        if found_file: break
    
    if not found_file:
        print("❌ Error: No .mat files found in the entire dataset!")
        return

    print(f"✅ Found file: {found_file}")
    
    # 2. Open it and print EVERYTHING
    try:
        mat = sio.loadmat(found_file)
        print("\n--- MATLAB FILE CONTENTS ---")
        print(f"Keys found: {list(mat.keys())}")
        
        # 3. Check for Data
        # We look for keys that don't start with '__' (which are just metadata)
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        
        if not data_keys:
            print("❌ Error: The MATLAB file contains no data variables!")
        else:
            for key in data_keys:
                data = mat[key]
                print(f"\nVariable: '{key}'")
                print(f"   Shape: {data.shape}")
                print(f"   Type:  {data.dtype}")
                
                # Check for NaNs or Infinite numbers
                if np.isnan(data).any():
                    print("   ⚠️ WARNING: Contains NaN values!")
                if np.isinf(data).any():
                    print("   ⚠️ WARNING: Contains Infinite values!")

    except Exception as e:
        print(f"\n❌ CRASHED while loading: {e}")

if __name__ == '__main__':
    main()