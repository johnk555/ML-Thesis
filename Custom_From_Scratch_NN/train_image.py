import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset_image import ImageDataset
from model_image import CustomImageModel
import time

# Τhe Control Center of τηε experiment τηατ orchestrates the entire learning process: loading data, feeding it to the model, calculating errors, and updating the brain.
# --- CONFIGURATION ---
ROOT_DIR = r'C:/Users/karel/Automotive' 
BATCH_SIZE = 32                                 # The model doesn't learn from 1 image at a time (too unstable) or all 16,000 at once (too big for memory). 
                                                #It learns in small chunks of 32. This provides a balance between training speed and gradient stability.

LEARNING_RATE = 0.001                           # This controls how "fast" the model changes its mind. 0.001: The "Goldilocks" number for the Adam optimizer.
EPOCHS = 15                                     # One "Epoch" means the model has seen the entire dataset once. Since we are training from scratch (random brain), we need more 
                                                # passes (15) compared to fine-tuning (which usually needs only 5-10) to allow the weights to converge.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Automatically detects my NVIDIA GPU. This accelerates the matrix math by ~50x compared to the CPU.

def main():
    print(f"--- Starting Custom Training (From Scratch) on {DEVICE} ---")
    
    # 1. Prepare Data Transforms
    # standard "Preprocessing Pipeline" for Convolutional Neural Networks (CNNs).
    # The number 224 wasn't chosen randomly. It is based on the architecture of Convolutional Neural Networks (CNNs).
    # Most modern CNNs (VGG, ResNet, EfficientNet) reduce the image size by half exactly 5 times. They use 5 Max-Pooling layers or Stride-2 Convolutions.
    # If you used a random number, say 250: 250 -> 125 -> 62.5 ... Crash.
    # Why its okay to throwing away some pixels (Original: 1440x1080 pixels -> Input: 224x224 pixels):
    # 1. Semantic Density: A car is still recognizable as a car even if it is small. You don't need 1080p resolution to tell a person from a tree.
    # 2. VRAM Limits: If you tried to train ResNet on full images, a single batch of 32 images would require 80 GB of Video Memory. Your GPU (6 GB) would crash instantly. 
    # 3. Focus: Since the objects are within 25 meters, are large enough in the frame that they survive the shrinking process. If you were trying to detect a pedestrian 
    # 200 meters away, resizing to 224 would make them disappear (they would become less than 1 pixel wide).
    transform = T.Compose([
        T.Resize((224, 224)),                   # Pooling layers (Input: 224, Pool 1: 112, Pool 2: 56, Pool 3: 28, Pool 4: 14)
        T.ToTensor(),                           # Pixels are 0 to 255 with shape (Height, Width, 3) and Pytorch needs 0 to 1 with shape (3, Height, Width)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        # These specific numbers are the average color (mean) and contrast (standard deviation) of the ImageNet dataset (14 million photos).
        # [0.485, 0.456, 0.406] = Average amount of Red, Green, and Blue in the world's photos.
        # keeping your data centered around zero helps the Optimizer (Adam) work faster.
    ])
    
    # 2. Load Dataset
    full_dataset = ImageDataset(ROOT_DIR, transform=transform)              # Instantiates your custom crawler that finds the pairs.
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size                               # Hide 20% of the data (val_data) from the model. The model never trains on this. We use it only
                                                                            # to test if the model is actually learning (generalizing) or just memorizing answers (overfitting).

    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training Samples: {train_size} | Validation Samples: {val_size}")
    
    # The automated worker. It grabs 32 images, stacks them into a single 4D Tensor ([32, 3, 224, 224]), and sends them to the GPU.
    # shuffle=True (Train): We shuffle the training data every epoch. If we didn't, the model might learn the order of images 
    # ("Oh, after the blue car comes the red truck") instead of the visual features.
    # shuffle=False (Val): We don't need to shuffle validation data because we are just measuring performance, not updating weights.
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Setup Model (Randomly Initialized)
    # CustomImageModel(): Creates an instance of the custom CNN. At this exact moment, the "brain" contains garbage—random numbers. It knows nothing.
    # .to(DEVICE): Physically moves the model's parameters (weights) into the VRAM of the GPU.
    model = CustomImageModel().to(DEVICE)
    
    # 4. Optimizer
    # optimizer (The Teacher):This is the algorithm that updates the brain. Adam (Adaptive Moment Estimation) is the industry standard because it adjusts the learning rate 
    # for each neuron individually.
    # criterion (The Grader): The Loss Function. BCEWithLogitsLoss: Binary Cross Entropy. It handles the math of "Multi-Label" problems.
    # It asks: "For the 'Car' neuron, did you output 1.0? If not, here is your error penalty." It does this for all 3 classes simultaneously.

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() 
    
    # 5. Training Loop (The engine)
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()                                                               # Switch to "Learning Mode" (enables Dropout)
        total_train_loss = 0
        
        # Phase A: --------------- Training ------------------- (The Study Session) 
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()                       # 1. Clear previous gradients. Deletes the "corrections" from the previous batch so they don't accumulate.
            outputs = model(images)                     # 2. Forward Pass (Guess)
            loss = criterion(outputs, labels)           # 3. Calculate Error
            loss.backward()                             # 4. Backward Pass (Calculate corrections). It uses Calculus (Chain Rule) to calculate exactly how much every 
                                                        # single weight in the network contributed to the error.
            optimizer.step()                  # 5. Update Weights (Apply corrections). Nudges every weight slightly in the opposite direction of the error to reduce it next time.
            total_train_loss += loss.item()

        # Phase B: --------------- Validation ------------------- (The Exam) 
        model.eval()                                    # Switch to "Evaluation Mode". It turns off layers like Dropout and BatchNorm that behave differently during training.
        total_val_loss = 0
        correct_preds = 0
        total_elements = 0
        
        with torch.no_grad():                           # Disable gradient calc (saves memory/speed). Tells PyTorch: "Don't track math for backpropagation. We are just looking."
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()           # Just counting total validation loss for the epoch.
                
                # Accuracy
                probs = torch.sigmoid(outputs)                                          # Convert logits to probabilities (0.0 to 1.0) using the Sigmoid function.
                preds = (probs > 0.5).float()                                           # Apply a threshold of 0.5 to get binary predictions. If prob > 0.5, predict 1.0 
                                                                                        # (object present); else 0.0 (object absent).

                correct_preds += (preds == labels).sum().item()                         # Counts how many predictions matched the ground truth labels.
                total_elements += labels.numel()                                        # Total number of labels (batch_size * 3)
        
        # Phase C: The Report Card (Stats)
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_preds / total_elements
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss                            # We only save the file if the model sets a new personal record on the validation set.
            # This ensures that if the model starts "Overfitting" (getting worse on new data) at Epoch 15, we still have the perfect version from Epoch 12 saved on the disk.
            torch.save(model.state_dict(), "custom_scratch_model.pth")
            print("  --> Model Saved (Improved Validation Loss)")

    print("\nTraining Complete. Best model saved as 'custom_scratch_model.pth'")

if __name__ == '__main__':
    main()