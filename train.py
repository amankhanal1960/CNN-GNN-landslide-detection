# --- TRAINING SCRIPT ----
import torch
import torch.nn as nn
from dataset import LandslideDataset

def train(img_dir, mask_dir, num_epochs=50, batch_size=16, lr=1e-4, save_path="best_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = LandslideDataset

