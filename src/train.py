import os
from src import ResUNet
import torch
from torch import optim
from torch.utils.data import DataLoader
from numpy import random
from src.data import LandslideDataset, compute_normalization, train_transform, val_transform
from src.utils import CombinedFocalDiceLoss, compute_metrics


def train(img_dir, mask_dir, num_epochs, batch_size, lr, save_path, graph_type="local", resume=False):

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
    
    train_dataset = LandslideDataset(img_dir=img_dir, mask_dir=mask_dir, transform=train_transform(MEAN, STD), file_ids=train_ids)
    val_dataset = LandslideDataset(img_dir=img_dir, mask_dir=mask_dir, transform=val_transform(MEAN, STD), file_ids=val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    # ----- Model, Loss and Optimizer ----- #
    model = ResUNet(in_channels=17, num_classes=2).to(device)
    criterion = CombinedFocalDiceLoss(focal_weight=0.35, dice_weight=0.65, alpha=0.85, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
    
    start_epoch = 1
    best_val_f1 = 0.0  
        
        # ----- Checkpoint Handling ----- #
    if resume:
        print(f"\n[INFO] Resuming training. Loading checkpoint from {save_path}...")
        checkpoint = torch.load(save_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']
        
        print(f"[INFO] Successfully resumed. Starting from epoch {start_epoch} (Previous Best F1: {best_val_f1:.4f})")
    else:
        print("\n[INFO] Starting training entirely from scratch.")

    # ----- Training Loop ----- #
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()            
            predictions = model(images)      
            loss = criterion(predictions, targets) 
            loss.backward()                  
            optimizer.step()                 
            
            running_train_loss += loss.item()
            
        train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
                
                batch_metrics = compute_metrics(predictions, targets)
                total_tp += batch_metrics[0]
                total_fp += batch_metrics[1]
                total_fn += batch_metrics[2]

        val_loss = running_val_loss / len(val_loader)
        
        val_metrics = {
            "iou": total_tp / (total_tp + total_fp + total_fn + 1e-6),
            "f1": 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6),
            "precision": total_tp / (total_tp + total_fp + 1e-6),
            "recall": total_tp / (total_tp + total_fn + 1e-6)
        }
        
        scheduler.step(1 - val_metrics['iou'])

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}  IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}  "
            f"Precision: {val_metrics['precision']:.4f}  Recall: {val_metrics['recall']:.4f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # ----- Strict Checkpoint Saving ----- #
        current_val_f1 = val_metrics['f1'] 
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1
            }
            torch.save(checkpoint, save_path)
            print(f" => Saved new best model checkpoint with F1: {best_val_f1:.4f}")
            
    print(f"\nTraining Complete! The best model achieved a Val F1 of {best_val_f1:.4f} and is saved at: {save_path}")
    return model