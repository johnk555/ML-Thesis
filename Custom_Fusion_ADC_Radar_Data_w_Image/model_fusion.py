import torch
import torch.nn as nn
from model_image import CustomImageModel 

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        
        # --- LEFT BRANCH: CAMERA ---
        # Initialize your custom image architecture
        self.image_branch = CustomImageModel()
        
        # Remove the 'classifier' (last layers). We only want features.
        # This keeps blocks 1-4. Output is 256x14x14
        self.image_features = nn.Sequential(
            self.image_branch.block1,
            self.image_branch.block2,
            self.image_branch.block3,
            self.image_branch.block4,
            nn.Flatten()
        )
        # Image Vector Size = 50176
        
        # --- RIGHT BRANCH: RADAR ---
        # Input: 1 Channel (Grayscale Map), 128x128
        self.radar_branch = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 32x32
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 16x16
            
            nn.Flatten()
        )
        # Radar Vector Size = 32*16*16 = 8192
        
        # --- FUSION HEAD ---
        # Concatenate: 50176 + 8192 = 58368
        self.fusion_head = nn.Sequential(
            nn.Linear(58368, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3) 
        )

    def forward(self, image, radar):
        # 1. Process Image
        img_out = self.image_features(image)
        
        # 2. Process Radar
        rad_out = self.radar_branch(radar)
        
        # 3. Concatenate
        combined = torch.cat((img_out, rad_out), dim=1)
        
        # 4. Final Prediction
        output = self.fusion_head(combined)
        return output