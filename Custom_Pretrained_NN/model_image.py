import torch
import torch.nn as nn
import torchvision.models as models

class CustomImageModel(nn.Module):
    def __init__(self):
        super(CustomImageModel, self).__init__()
        
        # 1. Load Pre-trained Backbone (ResNet18)
        # We use 'weights' instead of 'pretrained' to avoid warnings
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        
        # 2. FREEZE LAYERS (Transfer Learning)
        # We freeze all layers initially so we don't destroy the pre-trained features
        for param in backbone.parameters():
            param.requires_grad = False
            
        # 3. Create Custom Head
        # Remove the last fully connected layer (fc)
        # Input to fc is 512 for ResNet18
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Define new layers (We DO train these)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization to prevent overfitting
            nn.Linear(256, 3) # Output: 3 Classes [Person, Cyclist, Car]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x