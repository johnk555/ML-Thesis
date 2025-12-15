import os
import pandas as pd
from tqdm import tqdm

# Root directory
root_dir = r"C:\Users\karel\Automotive"
output_dir = os.path.join(root_dir, "YOLO_dataset")
os.makedirs(output_dir, exist_ok=True)

# Class mapping
label_map = {
    0: 'person',
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'truck',
    80: 'cyclist'
}

# Convert to continuous IDs for YOLO (0–5)
class_to_yolo = {k: i for i, k in enumerate(label_map.keys())}

# Image and physical ranges
IMG_W, IMG_H = 1440, 1080
X_RANGE, Y_RANGE = 40.0, 23.0

def convert_to_yolo(px, py, wid, leng):
    x_pixel = ((px + 20) / X_RANGE) * IMG_W
    y_pixel = ((24 - py) / Y_RANGE) * IMG_H
    w_pixel = (wid / X_RANGE) * IMG_W
    h_pixel = (leng / Y_RANGE) * IMG_H
    return (
        x_pixel / IMG_W,
        y_pixel / IMG_H,
        w_pixel / IMG_W,
        h_pixel / IMG_H
    )

for seq in tqdm(os.listdir(root_dir)):
    seq_path = os.path.join(root_dir, seq)
    if not os.path.isdir(seq_path):
        continue

    img_dir = os.path.join(seq_path, "images_0")
    lbl_dir = os.path.join(seq_path, "text_labels")
    if not (os.path.exists(img_dir) and os.path.exists(lbl_dir)):
        continue

    seq_output = os.path.join(output_dir, seq)
    os.makedirs(seq_output, exist_ok=True)

    for csv_file in os.listdir(lbl_dir):
        if not csv_file.endswith(".csv"):
            continue

        csv_path = os.path.join(lbl_dir, csv_file)
        frame_name = os.path.splitext(csv_file)[0]
        img_path = os.path.join(img_dir, f"{frame_name}.jpg")

        # Output label path (same name but .txt)
        out_label_path = os.path.join(seq_output, f"{frame_name}.txt")

        #df = pd.read_csv(csv_path, header=None, sep=",|\t", engine="python")

        # Skip empty files
        if os.path.getsize(csv_path) == 0:
            continue

        try:
            df = pd.read_csv(csv_path, header=None, sep=r"\s+|,|\t", engine="python")
        except pd.errors.EmptyDataError:
            continue
        # Columns: uid, class, px, py, wid, len
        with open(out_label_path, "w") as f:
            for _, row in df.iterrows():
                cls = int(row[1])
                if cls not in label_map:
                    continue
                cid = class_to_yolo[cls]
                px, py, wid, leng = map(float, row[2:6])
                x, y, w, h = convert_to_yolo(px, py, wid, leng)
                f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
