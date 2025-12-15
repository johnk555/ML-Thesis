import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset_image import ImageDataset
from model_image import CustomImageModel

# --- CONFIGURATION ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
MODEL_PATH = "custom_scratch_model.pth" # Points to your trained file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Loading Model for Evaluation on {DEVICE} ---")
    
    # 1. Setup Data
    # Use the exact same transform as training
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the full dataset (or you could split it, but testing on all is fine for a quick check)
    dataset = ImageDataset(ROOT_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Load the Model Architecture
    model = CustomImageModel().to(DEVICE)
    
    # 3. Load the Trained Weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_PATH}. Did training finish?")
        return

    model.eval()
    
    # 4. Initialize Counters
    # Map index to Name: 0=Person, 1=Cyclist, 2=Car
    class_names = {0: "Person", 1: "Cyclist", 2: "Car"}
    
    # Stats: {0: {'correct': 0, 'total': 0}, 1: ...}
    stats = {i: {'correct': 0, 'total': 0} for i in class_names.keys()}
    
    print("Running Inference...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get predictions
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Calculate Recall for each class
            for i in range(3): # Iterate over Person, Cyclist, Car
                # Get the column for this class
                class_preds = preds[:, i]
                class_labels = labels[:, i]
                
                # We only care when the Object was ACTUALLY there (Label = 1)
                # Find indices where ground truth says "Yes"
                true_indices = (class_labels == 1)
                
                # How many of those "Yes" instances did we get right?
                correct = (class_preds[true_indices] == 1).sum().item()
                total = true_indices.sum().item()
                
                stats[i]['correct'] += correct
                stats[i]['total'] += total

    # 5. Print Report
    print("\n" + "="*40)
    print("FINAL RECALL REPORT (Custom Model)")
    print("="*40)
    print(f"{'Class':<10} | {'Total Samples':<15} | {'Recall %':<10}")
    print("-" * 40)
    
    for i in range(3):
        name = class_names[i]
        total = stats[i]['total']
        correct = stats[i]['correct']
        
        if total > 0:
            recall = (correct / total) * 100
            print(f"{name:<10} | {total:<15} | {recall:.2f}%")
        else:
            print(f"{name:<10} | {0:<15} | N/A")
            
    print("="*40)

if __name__ == '__main__':
    main()