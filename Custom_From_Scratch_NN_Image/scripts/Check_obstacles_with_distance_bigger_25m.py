import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
DISTANCE_THRESHOLD = 25.0

# Define classes to track
CLASS_MAP = {
    0: 'Person',
    2: 'Car',
    80: 'Cyclist'
}

def main():
    print(f"Scanning for objects further than {DISTANCE_THRESHOLD} meters...")
    
    total_frames = 0
    frames_with_distant = 0
    
    # Stats counters
    total_objects = {name: 0 for name in CLASS_MAP.values()}
    distant_objects = {name: 0 for name in CLASS_MAP.values()}
    
    for root, dirs, files in os.walk(ROOT_DIR):
        if 'text_labels' in dirs:
            label_dir = os.path.join(root, 'text_labels')
            
            for file in os.listdir(label_dir):
                if file.endswith('.csv'):
                    total_frames += 1
                    csv_path = os.path.join(label_dir, file)
                    
                    try:
                        # Read CSV: [uid, class, px, py, wid, len]
                        df = pd.read_csv(csv_path, header=None)
                        
                        # Filter only classes we care about
                        df = df[df[1].isin(CLASS_MAP.keys())]
                        
                        if df.empty: continue
                        
                        # Check for distant objects in this frame
                        # Column 3 is 'py' (longitudinal distance)
                        is_distant_frame = False
                        
                        for _, row in df.iterrows():
                            cls_id = row[1]
                            dist = abs(row[3])
                            cls_name = CLASS_MAP[cls_id]
                            
                            # Count Total
                            total_objects[cls_name] += 1
                            
                            # Count Distant
                            if dist > DISTANCE_THRESHOLD:
                                distant_objects[cls_name] += 1
                                is_distant_frame = True
                        
                        if is_distant_frame:
                            frames_with_distant += 1
                            
                    except Exception:
                        continue

    # --- PRINT RESULTS ---
    print("\n" + "="*50)
    print(f"DISTANCE ANALYSIS (> {DISTANCE_THRESHOLD}m)")
    print("="*50)
    print(f"Total Frames Scanned: {total_frames}")
    print(f"Frames containing Distant Objects: {frames_with_distant}")
    print(f"Percentage of Frames affected: {(frames_with_distant/total_frames)*100:.2f}%")
    print("-" * 50)
    
    print(f"{'Class':<10} | {'Total Obj':<10} | {'> 25m Count':<12} | {'% Excluded':<10}")
    print("-" * 50)
    
    total_excluded = 0
    grand_total = 0
    
    for name in CLASS_MAP.values():
        tot = total_objects[name]
        dist = distant_objects[name]
        grand_total += tot
        total_excluded += dist
        
        if tot > 0:
            perc = (dist / tot) * 100
            print(f"{name:<10} | {tot:<10} | {dist:<12} | {perc:.2f}%")
        else:
            print(f"{name:<10} | {0:<10} | {0:<12} | N/A")
            
    print("-" * 50)
    print(f"TOTAL EXCLUDED OBJECTS: {total_excluded} / {grand_total}")
    print("="*50)

if __name__ == '__main__':
    main()