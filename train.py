import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from numpy import random
from src.dataset import LandslideDataset, train_transform, val_transform
from src.model import UNet
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
    
    train_dataset = LandslideDataset(img_dir=img_dir, mask_dir=mask_dir, transform=train_transform(), file_ids=train_ids)
    val_dataset = LandslideDataset(img_dir=img_dir, mask_dir=mask_dir, transform=val_transform(), file_ids=val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    # ----- Model, Loss and Optimizer ----- #
    model = UNet(in_channels=8, num_classes=2).to(device)
    criterion = CombinedFocalDiceLoss(focal_weight=0.5, dice_weight=0.5, alpha=0.75, gamma=2.0)
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
        
        # Lists to temporarily hold metrics across validation batches
        val_iou, val_f1, val_prec, val_rec = [], [], [], []
        
        with torch.no_grad(): # Disable gradient engine to save memory
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
                
                # Calculate metrics for this batch using your dictionary utility
                batch_metrics = compute_metrics(predictions, targets)
                val_iou.append(batch_metrics["iou"])
                val_f1.append(batch_metrics["f1"])
                val_prec.append(batch_metrics["precision"])
                val_rec.append(batch_metrics["recall"])

        val_loss = running_val_loss / len(val_loader)
        
        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Average the metrics over all validation batches
        val_metrics = {
            "iou": sum(val_iou) / len(val_iou),
            "f1": sum(val_f1) / len(val_f1),
            "precision": sum(val_prec) / len(val_prec),
            "recall": sum(val_rec) / len(val_rec)
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
    train(
        img_dir    = "E:\\Major project\\Datasets\\landslide4sense\\TrainData\\img",
        mask_dir   = "E:\\Major project\\Datasets\\landslide4sense\\TrainData\\mask",
        num_epochs = 50,
        batch_size = 16,
        lr         = 1e-4,
        save_path  = "best_unet_model.pth"
    )