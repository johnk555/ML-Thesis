import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import os
from torchvision import transforms
import torch.nn.functional as F

# --- CONFIGURATION ---
DATA_PATH = r'C:/Users/karel/Automotive'
IMAGE_MODEL_PATH = "custom_scratch_model.pth"
RADAR_MODEL_PATH = "best_radar_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
CLASSES = ['Person', 'Cyclist', 'Car']
CONFIDENCE_THRESHOLD = 0.8  # Flag predictions below this confidence

def evaluate_and_log_failures():
    print(f"--- Loading Models on {DEVICE} ---")
    
    # 1. Load Models
    from model_image import CustomImageModel 
    from model_radar import RadarModel
    
    try:
        model_img = CustomImageModel().to(DEVICE)
        model_rad = RadarModel().to(DEVICE)
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    print("Loading weights...")
    model_img.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model_rad.load_state_dict(torch.load(RADAR_MODEL_PATH, map_location=DEVICE, weights_only=False))
    
    model_img.eval()
    model_rad.eval()

    # 2. Load Dataset
    print(f"--- Scanning Dataset: {DATA_PATH} ---")
    from dataset_fusion import FusionDataset 
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = FusionDataset(DATA_PATH, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Data Loaded. Evaluating {len(test_dataset)} samples...")

    # 3. Storage
    all_targets = []
    all_preds_img = []
    all_preds_rad = []
    all_probs_img = []  # Store confidence scores
    all_probs_rad = []
    
    failed_samples = []
    suspicious_samples = []  # Both models agree but have low confidence

    # 4. Evaluation Loop
    print("--- Starting Evaluation ---")
    with torch.no_grad():
        for batch_idx, (images, radars, labels) in enumerate(test_loader):
            images, radars, labels = images.to(DEVICE), radars.to(DEVICE), labels.to(DEVICE)

            # Get logits and convert to probabilities
            logits_img = model_img(images)
            logits_rad = model_rad(radars)
            
            probs_img = F.softmax(logits_img, dim=1)
            probs_rad = F.softmax(logits_rad, dim=1)

            pred_img = torch.argmax(probs_img, dim=1)
            pred_rad = torch.argmax(probs_rad, dim=1)
            
            # Get confidence scores (max probability)
            conf_img, _ = torch.max(probs_img, dim=1)
            conf_rad, _ = torch.max(probs_rad, dim=1)

            # Handle labels (already class indices from dataset)
            if labels.ndim > 1 and labels.shape[1] > 1:
                labels_indices = torch.argmax(labels, dim=1)
            else:
                labels_indices = labels

            # Move to CPU
            batch_targets = labels_indices.cpu().numpy().flatten().astype(int)
            batch_preds_img = pred_img.cpu().numpy().flatten().astype(int)
            batch_preds_rad = pred_rad.cpu().numpy().flatten().astype(int)
            batch_conf_img = conf_img.cpu().numpy().flatten()
            batch_conf_rad = conf_rad.cpu().numpy().flatten()

            # Safety check
            if len(batch_targets) != len(batch_preds_img):
                print(f"⚠️  Batch {batch_idx}: size mismatch, skipping...")
                continue

            all_targets.extend(batch_targets)
            all_preds_img.extend(batch_preds_img)
            all_preds_rad.extend(batch_preds_rad)
            all_probs_img.extend(batch_conf_img)
            all_probs_rad.extend(batch_conf_rad)

            # Analyze failures and suspicious cases
            start_idx = batch_idx * BATCH_SIZE
            for i in range(len(batch_targets)):
                global_idx = start_idx + i
                
                img_correct = batch_preds_img[i] == batch_targets[i]
                rad_correct = batch_preds_rad[i] == batch_targets[i]
                
                img_conf = batch_conf_img[i]
                rad_conf = batch_conf_rad[i]

                try:
                    fname = os.path.basename(test_dataset.image_paths[global_idx])
                except:
                    fname = f"Sample_{global_idx}"

                # Case 1: At least one model failed
                if not img_correct or not rad_correct:
                    failed_samples.append({
                        "file": fname,
                        "truth": CLASSES[batch_targets[i]],
                        "img_pred": CLASSES[batch_preds_img[i]],
                        "rad_pred": CLASSES[batch_preds_rad[i]],
                        "img_conf": f"{img_conf:.2%}",
                        "rad_conf": f"{rad_conf:.2%}",
                        "status": "IMG_FAIL" if not img_correct else "RAD_FAIL" if not rad_correct else "BOTH_FAIL"
                    })
                
                # Case 2: Both correct but low confidence (possible labeling issue)
                elif img_correct and rad_correct and (img_conf < CONFIDENCE_THRESHOLD or rad_conf < CONFIDENCE_THRESHOLD):
                    suspicious_samples.append({
                        "file": fname,
                        "truth": CLASSES[batch_targets[i]],
                        "img_conf": f"{img_conf:.2%}",
                        "rad_conf": f"{rad_conf:.2%}",
                        "reason": "Low confidence despite correct prediction"
                    })

            if batch_idx % 10 == 0:
                print(f"Processed {min((batch_idx + 1) * BATCH_SIZE, len(test_dataset))}/{len(test_dataset)} samples...", end='\r')

    print("\n")
    
    # 5. Calculate Metrics
    img_accuracy = np.mean(np.array(all_targets) == np.array(all_preds_img)) * 100
    rad_accuracy = np.mean(np.array(all_targets) == np.array(all_preds_rad)) * 100
    avg_conf_img = np.mean(all_probs_img) * 100
    avg_conf_rad = np.mean(all_probs_rad) * 100

    print("="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Image Model Accuracy: {img_accuracy:.2f}% (Avg Confidence: {avg_conf_img:.2f}%)")
    print(f"Radar Model Accuracy: {rad_accuracy:.2f}% (Avg Confidence: {avg_conf_rad:.2f}%)")
    print(f"Total Samples: {len(all_targets)}")
    print(f"Failed Predictions: {len(failed_samples)}")
    print(f"Suspicious (Low Confidence): {len(suspicious_samples)}")

    # 6. Confusion Matrices
    print("\n" + "="*60)
    print("CONFUSION MATRIX - IMAGE MODEL")
    print("="*60)
    cm_img = confusion_matrix(all_targets, all_preds_img)
    print(f"{'':>10} " + " ".join([f"{c:>10}" for c in CLASSES]))
    for i, row in enumerate(cm_img):
        print(f"{CLASSES[i]:>10} " + " ".join([f"{val:>10}" for val in row]))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX - RADAR MODEL")
    print("="*60)
    cm_rad = confusion_matrix(all_targets, all_preds_rad)
    print(f"{'':>10} " + " ".join([f"{c:>10}" for c in CLASSES]))
    for i, row in enumerate(cm_rad):
        print(f"{CLASSES[i]:>10} " + " ".join([f"{val:>10}" for val in row]))

    # 7. Detailed Classification Report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT - IMAGE MODEL")
    print("="*60)
    print(classification_report(all_targets, all_preds_img, target_names=CLASSES, digits=4))
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT - RADAR MODEL")
    print("="*60)
    print(classification_report(all_targets, all_preds_rad, target_names=CLASSES, digits=4))

    # 8. Save Detailed Reports
    with open("failures_report.txt", "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FAILED PREDICTIONS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Total Failures: {len(failed_samples)}\n\n")
        
        # Group by failure type
        img_only = [s for s in failed_samples if s['status'] == 'IMG_FAIL']
        rad_only = [s for s in failed_samples if s['status'] == 'RAD_FAIL']
        both_fail = [s for s in failed_samples if s['status'] == 'BOTH_FAIL']
        
        f.write(f"Image Only Failed: {len(img_only)}\n")
        f.write(f"Radar Only Failed: {len(rad_only)}\n")
        f.write(f"Both Failed: {len(both_fail)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'Filename':<40} | {'Truth':<10} | {'Img Pred':<10} | {'Rad Pred':<10} | {'Img Conf':<10} | {'Rad Conf':<10}\n")
        f.write("-" * 80 + "\n")
        
        for fail in failed_samples:
            f.write(f"{fail['file']:<40} | {fail['truth']:<10} | {fail['img_pred']:<10} | {fail['rad_pred']:<10} | {fail['img_conf']:<10} | {fail['rad_conf']:<10}\n")
    
    with open("suspicious_report.txt", "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SUSPICIOUS SAMPLES (Low Confidence Despite Correct Prediction)\n")
        f.write("="*80 + "\n")
        f.write(f"Total Suspicious: {len(suspicious_samples)}\n")
        f.write("These samples might have labeling issues or ambiguous features.\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Filename':<40} | {'Truth':<10} | {'Img Conf':<10} | {'Rad Conf':<10}\n")
        f.write("-" * 80 + "\n")
        
        for susp in suspicious_samples:
            f.write(f"{susp['file']:<40} | {susp['truth']:<10} | {susp['img_conf']:<10} | {susp['rad_conf']:<10}\n")
    
    print(f"\n✅ Reports saved:")
    print(f"   - failures_report.txt ({len(failed_samples)} samples)")
    print(f"   - suspicious_report.txt ({len(suspicious_samples)} samples)")
    print("\n💡 Review 'suspicious_report.txt' for potential labeling errors!")

if __name__ == "__main__":
    evaluate_and_log_failures()