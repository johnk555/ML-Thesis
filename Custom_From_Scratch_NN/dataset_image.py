import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

# This class inherits from torch.utils.data.Dataset. In PyTorch, a Dataset class has two main jobs:
# 1. __init__ (The Architect): It runs once at the beginning. It crawls through your folders, finds all the valid files, and creates a master list (index).
# 2. __getitem__ (The Worker): It runs thousands of times during training. When the training loop asks for "Image #42", this method loads that specific file 
# from disk and converts it to numbers.
class ImageDataset(Dataset): # Defines the class. It inherits from Dataset, which gives it standard PyTorch superpowers (like compatibility with DataLoader).
    def __init__(self, root_dir, transform=None): # This section scans your hard drive to build the list of valid samples.
        # Constructor: It takes root_dir (where your data lives like C:\Users...) and transform (the resize/normalization logic).
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # This is an empty list that will act as the Registry. By the end of this function, it will look like: [("path/to/img1.jpg", "path/to/img1.csv"), ...]
        
        # --- CONFIGURATION ---
        # 0 -> Person, 1 -> Cyclist, 2 -> Car
        # Dataset class IDs to our simplified 3-class system
        # My NN has 3 specific output neurons: Neuron 0, Neuron 1, Neuron 2.
        # This dictionary maps the World ID to the Neuron ID. If the CSV says 80 (Cyclist), this map converts it to 1 so the second neuron lights up.
        self.target_map = {0: 0, 80: 1, 2: 2} 

        print("Scanning dataset for valid images and labels...")
        
        # A temporary counter. This is just for the information (so you can see "Found 10,880 Persons" printed in the terminal). It doesn't affect training.
        counts = {0: 0, 1: 0, 2: 0} 
        
        for seq in os.listdir(root_dir):                    # Starts a loop through every folder in C:\Users\karel\Automotive. Each seq is a folder like 2019_04_09_bms1000.
            seq_path = os.path.join(root_dir, seq)                                          # Creates the full path string.
            if not os.path.isdir(seq_path): continue                                        # If it's not a folder, and it is a file like readme.txt, skip it.
            
            img_dir = os.path.join(seq_path, 'images_0')                                    # Folder with images
            lbl_dir = os.path.join(seq_path, 'text_labels')                                 # Folder with CSV labels
            
            if not (os.path.exists(img_dir) and os.path.exists(lbl_dir)): continue          # If a folder is missing images_0, we skip this entire sequence to avoid errors.
            
            for f in os.listdir(img_dir):                                                   # Now we look inside the images_0 folder and loop through every file.
                if f.endswith('.jpg'):                                                      # We only care about JPG files, so if it's not a JPG, skip it.
                    csv_name = f.replace('.jpg', '.csv')                                    # If the image is 0000000001.jpg, we assume the label must be named 0000000001.csv.
                    csv_path = os.path.join(lbl_dir, csv_name)                              # Builds the full path to that CSV file.
                    
                    # This part ensures we don't train on empty or useless images.
                    if os.path.exists(csv_path):                                            # Checks if the label file actually exists.
                        try:
                            df = pd.read_csv(csv_path, header=None)              # Uses pandas to open the csv and header=None tells pandas that row 0 is data, not column names.
                            valid_rows = df[ (df[1].isin(self.target_map.keys()))]          # df[1]:Look at second column (Column 1) which is the Class ID Column. 
                                                                                            # isin(self.target_map.keys()): Checks if the class ID is 0, 2, or 80.
                                                                                        # Returns a smaller DataFrame with only the rows we care about. (Ignore Truck, Bus, etc)
                            
                            if not valid_rows.empty:                        # If valid_rows is not empty, it means this image contains at least one object we care about.
                                self.samples.append((os.path.join(img_dir, f), csv_path))   # We add this image/label pair to our master list self.samples.
                                for cls_id in valid_rows[1].unique():
                                    counts[self.target_map[cls_id]] += 1                    # Updates our statistics counter so you get that nice printout at the end.
                        except:
                            continue

        print(f"Dataset Ready. Found {len(self.samples)} valid frames.")
        print(f"Distribution: Person: {counts[0]}, Cyclist: {counts[1]}, Car: {counts[2]}")

    # This method is required by PyTorch so the DataLoader knows how big the dataset is (e.g., to calculate how many batches are in one epoch).
    def __len__(self):
        return len(self.samples)                                                            # Returns the total number of valid samples found during initialization.

    # This is the engine of your data pipeline. 
    # It runs every time the training loop asks for data. It takes an index (idx), grabs that specific file from the hard drive, transforms it, and returns the tensors.
    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]                                # Unpacks the tuple into image path and csv path. (etc. [("path/img.jpg", "path/data.csv"), ...].)
        
        # 1. Load Image
        # Uses the Pillow (PIL) library to load the image file from the disk into memory.
        # Convertes the image to RGB format. Sometimes JPGs are saved as CMYK or Grayscale; this forces them into the standard RGB format your model expects.
        image = Image.open(img_path).convert('RGB')                  

        if self.transform:                  # Checks if you defined a transform in __init__ (the Resize/ToTensor logic we discussed earlier).
            image = self.transform(image)   # If so, the variable image changes from a PIL Image (0-255 integer pixels) to a PyTorch Tensor (0.0-1.0 float numbers, 3x224x224).
        
        # 2. Load Label
        # Creates a tensor of 3 zeros: [0.0, 0.0, 0.0]. This is our multi-label target vector (Person, Cyclist, Car).
        # We start with "empty" (assuming nothing is in the image) and will flip specific bits to 1.0 if we find the object in the CSV.
        label = torch.zeros(3, dtype=torch.float32) 

        try:
            df = pd.read_csv(csv_path, header=None)                     # Loads the text label file into a Pandas DataFrame.
            valid_rows = df[ (df[1].isin(self.target_map.keys())) ]     # Keeps only rows where Column 1 (Class ID) is 0, 2, or 80.
            for cls_id in valid_rows[1].unique(): # Loops through every unique object class found in the valid rows. If the image has 3 Cars and 1 Person. unique() returns [2, 0]
                idx = self.target_map[cls_id]   # self.target_map[cls_id]: Translates the dataset ID to the neuron index. 2(Car) -> 2, 0(Person) -> 0
                label[idx] = 1.0            # Turns that neuron "ON". Final label becomes: [1.0, 0.0, 1.0] (Person=Yes, Cyclist=No, Car=Yes).
        except: # If a CSV file is corrupted, empty, or unreadable, the code won't crash the entire training run and jump to except and return the empty [0, 0, 0] label.
            pass 

        return image, label