import os

# --- UPDATE THIS TO MATCH YOUR ROOT DIR ---
ROOT_DIR = r'C:/Users/karel/Automotive' 

def check_structure():
    print(f"Checking ROOT_DIR: {ROOT_DIR}")
    
    if not os.path.exists(ROOT_DIR):
        print("❌ ERROR: Root directory does not exist!")
        return

    sequences = os.listdir(ROOT_DIR)
    print(f"Found {len(sequences)} items in root.")
    
    # Check the first valid sequence folder
    found_seq = False
    for seq in sequences:
        seq_path = os.path.join(ROOT_DIR, seq)
        if not os.path.isdir(seq_path): continue
        
        found_seq = True
        print(f"\n--- Checking Sequence: {seq} ---")
        
        img_dir = os.path.join(seq_path, 'images_0')
        radar_dir = os.path.join(seq_path, 'radar_raw_frame')
        lbl_dir = os.path.join(seq_path, 'text_labels')
        
        # Check Folders
        print(f"1. Checking Image Folder: {img_dir} ... {'✅ Found' if os.path.exists(img_dir) else '❌ MISSING'}")
        print(f"2. Checking Label Folder: {lbl_dir} ... {'✅ Found' if os.path.exists(lbl_dir) else '❌ MISSING'}")
        print(f"3. Checking Radar Folder: {radar_dir} ... {'✅ Found' if os.path.exists(radar_dir) else '❌ MISSING'}")
        
        if not os.path.exists(radar_dir):
            print("   -> ⚠️ CRITICAL: Radar folder is missing. Do you have the radar data downloaded?")
            break
            
        # Check File Matching
        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        radars = [f for f in os.listdir(radar_dir) if f.endswith('.mat')]
        
        print(f"   -> Found {len(images)} Images.")
        print(f"   -> Found {len(radars)} Radar MAT files.")
        
        if len(images) > 0 and len(radars) == 0:
            print("   -> ❌ ERROR: Radar folder exists but is empty (or has no .mat files).")
        
        if len(images) > 0 and len(radars) > 0:
            # Check pairing logic
            sample_img = images[0]
            expected_radar = sample_img.replace('.jpg', '.mat')
            if expected_radar in radars:
                print(f"   -> ✅ SUCCESS: Found matching pair! ({sample_img} <-> {expected_radar})")
            else:
                print(f"   -> ❌ ERROR: Name mismatch.")
                print(f"      Image name: {sample_img}")
                print(f"      Expected Radar: {expected_radar}")
                print(f"      Actual Radar file example: {radars[0]}")
        
        break # Only check the first folder to keep output clean

    if not found_seq:
        print("❌ ERROR: No sequence folders found inside Root.")

if __name__ == '__main__':
    check_structure()