import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path

class WaterScenesDataset(Dataset):
    """
    Custom PyTorch Dataset for the WaterScenes dataset.
    """
    def __init__(self, root_dir, split_file, image_transform=None, target_transform=None,
                 radar_mean=None, radar_std=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.target_transform = target_transform

        if radar_mean and radar_std:
            # Shape [4, 1, 1] for broadcasting against [4, H, W]
            self.radar_mean = torch.tensor(radar_mean, dtype=torch.float32).view(4, 1, 1)
            self.radar_std = torch.tensor(radar_std, dtype=torch.float32).view(4, 1, 1)
        else:
            self.radar_mean = None
            self.radar_std = None
        
        # Define file paths
        self.image_dir = os.path.join(root_dir, 'image')
        self.radar_revp_dir = os.path.join(root_dir, 'radar_revp_npy')
        self.label_dir = os.path.join(root_dir, 'detection', 'yolo')

        # Load the file IDs from the split file
        self.file_ids = self._load_file_ids(split_file)

        # Shuffle and Slice for Subset (if training)
        if 'train' in str(split_file):
            # Use a fixed seed for reproducibility
            random.seed(42) 
            random.shuffle(self.file_ids)
            
            # Select percentage of data to use
            subset_ratio = 1.0 
            subset_size = int(len(self.file_ids) * subset_ratio)
            
            # Slice the list
            self.file_ids = self.file_ids[:subset_size]
            print(f"Training subset ({int(subset_ratio*100)}%): {len(self.file_ids)} samples")

        # Add a check for the preprocessed folder existence
        if not os.path.exists(self.radar_revp_dir):
            raise FileNotFoundError(
                f"Preprocessed radar directory not found: {self.radar_revp_dir}\n"
                "Please run the `preprocess_data.py` script first."
            )

    def _load_file_ids(self, split_file_path):
        """Helper function to read the .txt file and return a list of IDs."""
        with open(split_file_path, 'r') as f:
            # Process each line
            file_ids = []
            for line in f:
                line = line.strip()
                if line:
                    file_id = Path(line).stem
                    file_ids.append(file_id)
        return file_ids

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_ids)

    def __getitem__(self, idx):
        # Get the file ID for this index
        file_id = self.file_ids[idx]
        
        # Load Image (unecessary for now)
        """
        img_path = os.path.join(self.image_dir, f"{file_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        w, h = image.size 
        original_image_size = (h, w) # Pass (H, W) tuple
        """
        image = torch.zeros(1)

        # Load 4D Radar Data (REVP pre-processed .npy file)
        radar_path = os.path.join(self.radar_revp_dir, f"{file_id}.npy")
        try:
            radar_revp_numpy = np.load(radar_path)
            radar_tensor = torch.tensor(radar_revp_numpy, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading preprocessed radar file {radar_path}: {e}")
            # Fallback to an empty tensor with correct shape (C, H, W)
            C, H, W = 4, 320, 320 
            radar_tensor = torch.zeros((C, H, W), dtype=torch.float32)

        if self.radar_mean is not None:
            # Formula: (x - mean) / std
            # Add epsilon to std to avoid division by zero
            radar_tensor = (radar_tensor - self.radar_mean) / (self.radar_std + 1e-6)

        # Load Label (YOLO format)
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        labels = []
        
        # Scale to image size
        IMG_H, IMG_W = 320, 320 
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = float(parts[0])
                            
                            # Filter Classes (0-6)
                            if cls_id < 0 or cls_id >= 7: 
                                continue
                            
                            # Parse Normalized Coordinates (0.0 - 1.0)
                            n_cx, n_cy, n_w, n_h = [float(p) for p in parts[1:5]]
                            
                            # Convert to Absolute Pixel Coordinates
                            abs_cx = n_cx * IMG_W
                            abs_cy = n_cy * IMG_H
                            abs_w  = n_w  * IMG_W
                            abs_h  = n_h  * IMG_H
                            
                            labels.append([cls_id, abs_cx, abs_cy, abs_w, abs_h])
                            
            except Exception as e:
                print(f"Error loading label file {label_path}: {e}")

        """
        # --- AUGMENTATION: Random Horizontal Flip ---
        # 50% chance to flip
        if torch.rand(1) < 0.5:
            # Flip Radar (Shape: 4, H, W). Flip on last dim (Width)
            radar_tensor = torch.flip(radar_tensor, dims=[-1])
            
            # Flip Box Coordinates
            # YOLO Format: [class, x_center, y_center, w, h] (Absolute Pixels)
            if len(labels) > 0:
                # New X_center = Image_Width - Old_X_center
                label_tensor[:, 1] = IMG_W - label_tensor[:, 1]
        """
        
        # Convert to a tensor
        # If no objects, create an empty tensor of the correct shape [0, 5]
        if not labels:
            label_tensor = torch.empty((0, 5), dtype=torch.float32)
        else:
            label_tensor = torch.tensor(labels, dtype=torch.float32)

        # Apply Transforms 
        if self.image_transform:
            #image = self.image_transform(image)
            None
        if self.target_transform:
            # target_transform now operates on a [N_objects, 5] tensor
            label_tensor = self.target_transform(label_tensor)

        # Return data as a dictionary
        sample = {
            'image': image,
            'radar': radar_tensor,
            'label': label_tensor, # Shape: [N_objects, 5]
            'file_id': file_id
        }
        
        return sample
    
def collate_fn(batch):
    """
    Custom collate function to handle variable-length radar data
    and variable-length YOLO labels.
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Radar data is a list of tensors (one for each item in the batch)
    radar_data = torch.stack([item['radar'] for item in batch])   

    #Collate file ids
    file_ids = [item['file_id'] for item in batch]
     
    # Process labels (YOLO format)
    labels = []
    for i, item in enumerate(batch):
        label = item['label']  # This is [N_objects, 5]
        
        # Check if there are any objects in this sample
        if label.shape[0] > 0:
            # Create a tensor for the batch index, shape [N_objects, 1]
            batch_idx_col = torch.full((label.shape[0], 1), i)
            
            # Prepend batch index: [batch_idx, class_id, x, y, w, h]
            label_with_idx = torch.cat((batch_idx_col, label), dim=1)
            labels.append(label_with_idx)
    
    # Concatenate all labels from all batch items into one tensor
    if labels:
        labels_tensor = torch.cat(labels, 0)
    else:
        # No objects in this entire batch
        labels_tensor = torch.empty((0, 6), dtype=torch.float32)
    
    return {
        'image': images,
        'radar': radar_data,
        'label': labels_tensor,  # Shape: [Total_Objects_In_Batch, 6]
        'file_ids': file_ids
    }