import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. CONFIGURATION
ROOT_DIR = r'C:\Users\karel\Automotive' 

# Mapping based on README
ID_MAP = {
    0: 'Person',
    2: 'Car',
    3: 'Motorbike',
    5: 'Bus',
    7: 'Truck',
    80: 'Cyclist'
}

# 2. COUNTER
class_counts = {name: 0 for name in ID_MAP.values()}
total_frames = 0
frames_with_objects = 0

print(f"Scanning labels in {ROOT_DIR}...")

for root, dirs, files in os.walk(ROOT_DIR):
    if 'text_labels' in dirs:
        label_dir = os.path.join(root, 'text_labels')
        
        for file in os.listdir(label_dir):
            if file.endswith('.csv'):
                total_frames += 1
                csv_path = os.path.join(label_dir, file)
                
                try:
                    # Read CSV (Column 1 is Class ID)
                    df = pd.read_csv(csv_path, header=None)
                    
                    # Get unique objects in this frame
                    unique_ids = df[1].unique()
                    
                    if len(unique_ids) > 0:
                        frames_with_objects += 1
                    
                    # Count them
                    for uid in unique_ids:
                        if uid in ID_MAP:
                            name = ID_MAP[uid]
                            class_counts[name] += 1
                        else:
                            # Log unknown IDs if any exist
                            unknown_key = f"Unknown_ID_{uid}"
                            class_counts[unknown_key] = class_counts.get(unknown_key, 0) + 1
                            
                except Exception:
                    continue # Skip empty/bad CSVs

# 3. PRINT RESULTS
print("\n" + "="*30)
print("DATASET CLASS DISTRIBUTION")
print("="*30)
print(f"Total Frames Scanned: {total_frames}")
print("-" * 30)
print(f"{'CLASS':<15} | {'COUNT (Frames)'}")
print("-" * 30)

sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

for name, count in sorted_counts:
    print(f"{name:<15} | {count}")

print("="*30)

# 4. OPTIONAL: VISUALIZE
names = [x[0] for x in sorted_counts if x[1] > 0]
values = [x[1] for x in sorted_counts if x[1] > 0]

plt.figure(figsize=(10, 6))
plt.bar(names, values, color='skyblue')
plt.title('Class Distribution in Dataset')
plt.xlabel('Class')
plt.ylabel('Number of Frames appearing')
plt.show()