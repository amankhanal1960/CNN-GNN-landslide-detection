import os
from src import ResUNet
import torch
from torch import optim
from torch.utils.data import DataLoader
from numpy import random
from src.data import LandslideDataset, compute_normalization, train_transform, val_transform
from src.UNet_only import UNet
from src.utils import CombinedFocalDiceLoss, compute_metrics


def train(img_dir, mask_dir, num_epochs, batch_size, lr, save_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # extracting and shuffling the file IDs
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".h5")])
    all_ids = [int(f.split("_")[1].split(".")[0]) for f in all_files]
    
    # Shuffle with a fixed seed so train and val stays consistent
    random.seed(42)
    random.shuffle(all_ids)
    
    train_size = int(0.85 * len(all_ids))
    train_ids = all_ids[:train_size]
    val_ids = all_ids[train_size:]
    
    print(f"Total Samples: {len(all_ids)} | Train: {len(train_ids)} | Val: {len(val_ids)}")
    
    # Computing the mean and the standard deviations
    MEAN, STD = compute_normalization(img_dir, train_ids)
    
    train_dataset = LandslideDataset(img_dir=img_dir, mask_dir=mask_dir, transform=train_transform(), file_ids=train_ids, mean=MEAN, std=STD)
    val_dataset = LandslideDataset(img_dir=img_dir, mask_dir=mask_dir, transform=val_transform(), file_ids=val_ids, mean=MEAN, std=STD)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    # ----- Model, Loss and Optimizer ----- #
    model = ResUNet(in_channels=18, num_classes=2).to(device)
    criterion = CombinedFocalDiceLoss(focal_weight=0.4, dice_weight=0.6, alpha=0.85, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
    
    for epoch in range(1, num_epochs + 1):
        
        model.train()
        running_train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()            # Clear old gradients
            predictions = model(images)      # Forward pass
            loss = criterion(predictions, targets) # Calculate loss
            loss.backward()                  # Backward pass (gradients)
            optimizer.step()                 # Update model weights
            
            running_train_loss += loss.item()
            
        train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        with torch.no_grad(): # Disable gradient engine to save memory
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
                
                # Calculate metrics for this batch using your dictionary utility
                batch_metrics = compute_metrics(predictions, targets)
                total_tp += batch_metrics[0]
                total_fp += batch_metrics[1]
                total_fn += batch_metrics[2]

        val_loss = running_val_loss / len(val_loader)
        
        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Average the metrics over all validation batches
        val_metrics = {
            "iou": total_tp / (total_tp + total_fp + total_fn + 1e-6),
            "f1": 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6),
            "precision": total_tp / (total_tp + total_fp + 1e-6),
            "recall": total_tp / (total_tp + total_fn + 1e-6)
        }

        # ---- Print Progress ----- #
        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}  IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}  "
            f"Precision: {val_metrics['precision']:.4f}  Recall: {val_metrics['recall']:.4f}"
        )
        
    # Save the final trained model weights to your E: drive or local path
    torch.save(model.state_dict(), save_path)
    print("Training Complete! Model Saved.")
    
    return model


if __name__ == "__main__":
    # 1. Corrected Cloud Paths (Updated with the landslide4sense parent folder)
    IMG_DIR  = "/content/landslide4sense/TrainData/img"
    MASK_DIR = "/content/landslide4sense/TrainData/mask"
    
    # 2. Permanent Cloud Storage (Saves the model weights directly to your Google Drive)
    SAVE_PATH = "/content/drive/MyDrive/best_unet_model.pth"

    # 3. Hyperparameters
    BATCH_SIZE = 16  
    EPOCHS     = 65
    LEARNING_RATE = 1e-4

    print("--- Starting Cloud Landslide Mapping Pipeline ---")
    
    # 4. Kick off training
    trained_model = train(
        img_dir    = IMG_DIR,
        mask_dir   = MASK_DIR,
        num_epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        lr         = LEARNING_RATE,
        save_path  = SAVE_PATH
    )