import os
import shutil
import pandas as pd

# CONFIG
ROOT_DIR = r'C:\Users\karel\Automotive'
OUTPUT_DIR = r'C:\Users\karel\Desktop\Check_ID_1' # We will copy images here
TARGET_ID = 1.0 # The mystery ID

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Hunting for Class ID {TARGET_ID}...")
found_count = 0

for root, dirs, files in os.walk(ROOT_DIR):
    if 'text_labels' in dirs:
        label_dir = os.path.join(root, 'text_labels')
        image_dir = os.path.join(root, 'images_0')
        
        for file in os.listdir(label_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(label_dir, file)
                
                try:
                    df = pd.read_csv(csv_path, header=None)
                    # Check if ID 1 is in this file (Column 1)
                    if TARGET_ID in df[1].values:
                        
                        # We found one! Copy the corresponding image.
                        img_name = file.replace('.csv', '.jpg')
                        src_img = os.path.join(image_dir, img_name)
                        dst_img = os.path.join(OUTPUT_DIR, f"Mystery_{found_count}_{img_name}")
                        
                        if os.path.exists(src_img):
                            shutil.copy(src_img, dst_img)
                            print(f"Found one! Copied to {dst_img}")
                            found_count += 1
                            
                        if found_count >= 10: # Stop after finding 10 examples
                            print("Found 10 examples. Stopping.")
                            exit()
                            
                except Exception:
                    continue