import torch
import torch.nn as nn

# This class defines a Convolutional Neural Network (CNN) designed from scratch.
# It follows a classic "VGG-style" architecture: a sequential stack of convolutional blocks that progressively downsample the spatial dimensions (height/width) 
# while increasing the depth (number of feature channels). The network acts as a funnel, compressing the raw image into a dense feature vector for classification.
# Inherits from PyTorch's base NN class and provides all the necessary mechanisms for tracking gradients (backpropagation) and managing parameters.
class CustomImageModel(nn.Module):
    def __init__(self):                                             # This method defines the layers and structure of the model. 
        super(CustomImageModel, self).__init__()                    # Initializes the parent class. This is mandatory in PyTorch to register the layers properly.
        
        # --- CUSTOM ARCHITECTURE (From Scratch) ---
        # The network is divided into 4 identical "blocks." Each block performs specific operations to extract features.
        
        # Input: 3 Channels (RGB), 224x224
        # BLOCK 1: Edges & Colors (3 -> 32 channels)
        # The Convolutional layers (nn.Conv2d) do not need to know the image size (224). They only need to know the "Depth" (Channels).
        # However, the Linear layer at the end DOES need to know the size.
        # Imagine you have a small sticker (3*3 Kernel). Your job is to slide this sticker over a poster (the Image).
        # If the poster is small (224*224): You slide the sticker a few times. If the poster is huge (1000*1000): You slide the sticker many times.
        # In both cases, the sticker itself (3*3 kernel) does not change. The mathematical operation is identical regardless of the image width.
        # This is why nn.Conv2d only asks for in_channels (3 for RGB) and out_channels (32 filters), but never height or width.

        self.block1 = nn.Sequential(                                # A container that groups layers together. Data flows through them in order.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),             # The Convolutional Layer: 
                                                                    # 3 (Input Channels): Corresponds to the RGB channels of the input image.
                                                                    # 32 (Output Channels): We create 32 distinct filters. Each filter learns to recognize a simple pattern 
                                                                    # (e.g., a vertical edge, a specific color gradient).
                                                                    # Why start with 32? This is a fundamental design pattern in Deep Learning called the "Pyramid Architecture."
                                                                    # Almost every successful CNN (VGG, ResNet, YOLO) follows this exact pattern.
                                                                    # The first layer looks at the raw pixels. Its job is to find Low-Level Features: simple lines, edges, 
                                                                    # corners, and color gradients.
                                                                    # 1. Limited Variety: There are only so many ways to draw a line (horizontal, vertical, diagonal) 
                                                                    # or a color blob. You do not need 1000 filters to describe these simple patterns.
                                                                    # 2. Research has shown that 32 filters are sufficient to capture all the basic building blocks of an image.
                                                                    # 3. Resolution Cost: The first layer operates on the full image size (224*224). If you used 512 filters
                                                                    # here, the memory usage would explode because every single pixel would need 512 values attached to it.


                                                                    # kernel_size=3: Uses a standard 3*3 sliding window.
                                                                    # padding=1: Adds a border of zeros around the image. This preserves the spatial size (224*224) after 
                                                                    # the convolution,so we only lose size during pooling.

            nn.BatchNorm2d(32),                                     # Batch Normalization. It normalizes the output of the convolution (subtracts mean, divides by variance).
                                                                    # This is critical when training from scratch. It stabilizes learning, allows higher learning rates, 
                                                                    # and prevents gradients from vanishing.

            nn.ReLU(),                                  # Activation Function (Rectified Linear Unit). It turns negative values to zero (f(x) = max(0, x)). This  
                                                        # introduces non-linearity, allowing the model to learn complex shapes rather than just linear combinations of pixels.

            nn.MaxPool2d(2)                             # Output: 112x112     
                                                        # Downsampling. It takes the maximum value in every 2*2 window.  
                                                        # Effect: Halves the image size (224->112). 
                                                        # This reduces computation and makes the model "translation invariant" (it recognizes a car even if it shifts slightly).
        )
        
        # Block 2, 3, 4: High-Level Features
        # Channel Doubling (32->64->128->256): As you move deeper into the network, the layers combine the simple features from the previous step.

        # Why: Deeper layers detect more complex combinations. 
        # Block 1 (32): Detects lines or edges.
        # Block 2 (64): Combines lines to make Textures (grids, curves).
        # Block 3 (128): Combines textures to make Parts (wheels, windows, heads).
        # Block 4 (256): Combines parts to make Objects (Cars, Cyclists) and see "wheel shapes" or "head-and-shoulder" patterns. 
        # We need more filters to represent these complex combinations.

        # Spatial Reduction (112->56->28->14): We aggressively shrink the image size. By Block 4, the image is only 14*14 pixels, 
        # but each "pixel" holds a massive amount of semantic information (256 features).
        # This is the most mathematical reason. You are trading Space for Depth. 
        # Every time you run MaxPool, you destroy 75% of your spatial pixels (width and height get cut in half, so area becomes 1/4).
        # If you didn't increase channels: You would be throwing away information and making the network "dumber" at every step.
        # By doubling channels: You compensate for the spatial loss. You squeeze the spatial information into the depth dimension.

        # BLOCK 2: Textures (32 -> 64 channels)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output: 56x56
        )
        
        # BLOCK 3: Complex Shapes (64 -> 128 channels)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output: 28x28
        )
        
        # BLOCK 4: Objects (128 -> 256 channels)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output: 14x14
        )
        
        # CLASSIFIER: Decision Making
        # After extracting features, we need to make a final prediction.
        # While the Convolution layers don't care, the Fully Connected (Linear) Layer cares deeply.

        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # Transforms the 3D feature map (Shape: [Batch, 256, 14, 14]) into a 1D vector (Shape: [Batch, 50176]). 
                                                            # This prepares the data for the fully connected layers.

            nn.Linear(256 * 14 * 14, 512),                  # The Dense Layer (Fully Connected Layer). It connects every input feature to 512 hidden neurons. 
                                                            # The input size calculation is mathematically strict: 256 (channels) * 14 (height) * 14 (width) = 50,176.
                                                            # IF tried to feed a 300*300 image into this network:
                                                            # 1. The Conv layers would happily process it.
                                                            # 2. The output at Block 4 would be approx 18*18.
                                                            # 3. The Flatten layer would produce a vector of size 256*18*18 = 82,944.
                                                            # 4. This would NOT match the expected input size of 50,176 for the Linear layer, causing a runtime error.

            nn.ReLU(),
            nn.Dropout(0.5),                                # Regularization: During training, this randomly turns off 50% of the neurons. This prevents the model from relying 
                                                            # too heavily on any single neuron, forcing it to learn more robust, distributed representations. 
                                                            # This is crucial when training from scratch to avoid overfitting.

            nn.Linear(512, 3)                               # The final projection layer. It outputs 3 raw scores (logits), one for each class: Person, Cyclist, Car.
        )

# This method defines how data flows through the initialized layers.
# The code sequentially passes x through each block, overwriting x with the result.
# x: The input tensor batch (Images)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x