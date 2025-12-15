import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load image
img_path = 'C:/Users/karel/Automotive/2019_04_30_pbms002/images_0/0000000086.jpg'
img = cv2.imread(img_path)

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img)
plt.show()

# Perform inference
results = model(img)

# Print results
results.print()  # Print results to console
results.show()   # Display results

# Get the results
detections = results.pandas().xyxy[0]  # Results as pandas dataframe

# Print detections
print(detections)