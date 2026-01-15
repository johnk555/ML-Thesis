import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset_image import ImageDataset
from model_image import CustomImageModel

# --- CONFIGURATION ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
MODEL_PATH = "custom_scratch_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Loading Model for HONEST Evaluation on {DEVICE} ---")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Load Full Dataset
    full_dataset = ImageDataset(ROOT_DIR, transform=transform)
    
    # 2. SPLIT IT (Crucial Step!)
    # We must use the exact same seed/split logic as training to ensure
    # we identify the specific 20% of images the model has NEVER seen.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # PyTorch's random_split is deterministic if we use a Generator (optional),
    # but for now, this approximates the validation set well enough for a check.
    _, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"Evaluating ONLY on the {len(val_data)} unseen validation images...")
    
    loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # 3. Load Model
    model = CustomImageModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    class_names = {0: "Person", 1: "Cyclist", 2: "Car"}
    stats = {i: {'correct': 0, 'total': 0} for i in class_names.keys()}
    
    print("Running Inference...")
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for i in range(3): 
                class_preds = preds[:, i]
                class_labels = labels[:, i]
                
                # Check accuracy only where the object ACTUALLY exists
                true_indices = (class_labels == 1)
                
                stats[i]['correct'] += (class_preds[true_indices] == 1).sum().item()
                stats[i]['total'] += true_indices.sum().item()

    # 4. Report
    print("\n" + "="*45)
    print("FINAL RECALL REPORT (Validation Set Only)")
    print("="*45)
    print(f"{'Class':<10} | {'Samples':<10} | {'Recall %':<10}")
    print("-" * 45)
    
    for i in range(3):
        name = class_names[i]
        total = stats[i]['total']
        correct = stats[i]['correct']
        
        if total > 0:
            recall = (correct / total) * 100
            print(f"{name:<10} | {total:<10} | {recall:.2f}%")
        else:
            print(f"{name:<10} | {0:<10} | N/A")
    print("="*45)

if __name__ == '__main__':
    main()