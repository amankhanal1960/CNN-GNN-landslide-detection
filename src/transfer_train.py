import os
from ee import image
import torch
import random
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import compute_normalization, train_transform, val_transform, LandslideDataset
from src import ResUNet
from src.utils import compute_metrics
from src.utils import CombinedFocalDiceLoss, compute_metrics


def train_transfer_learning(
    train_img_dir,
    train_mask_dir, 
    val_img_dir,
    val_mask_dir,
    phase1_epochs,
    phase2_epochs,
    batch_size, 
    pretrained_model_path,
    save_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for the trainng!!!")
    
    train_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith(".h5")])
    train_ids = [int(f.split("_")[1].split(".")[0]) for f in train_files]
    
    
    val_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(".h5")])
    all_val_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    random.seed(42)
    random.shuffle(all_val_ids)
    
    split_idx = len(all_val_ids) // 2
    val_ids = all_val_ids[:split_idx]
    test_ids = all_val_ids[split_idx:]
    
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    
    print("\n ---- Computing Normalization Statistics From Nepal Training Split ----")
    MEANS, STDS = compute_normalization(train_img_dir, train_ids)
    print(f"Computed Means: {MEANS}")
    print(f"Computed Stds: {STDS}\n")
    
    # Initializing Datasets
    train_dataset = LandslideDataset(train_img_dir, train_mask_dir, transform=train_transform(MEANS, STDS), file_ids=train_ids)
    val_dataset = LandslideDataset(val_img_dir, val_mask_dir, transform=val_transform(MEANS, STDS), file_ids=val_ids)    
    test_dataset =  LandslideDataset(val_img_dir, val_mask_dir, transform=val_transform(MEANS, STDS), file_ids=test_ids)
        
     # Initializing Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)   
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)   
    
    # -------------- MODEL, LOSS & PRETRAINED WEIGHTS ----------------- #
                        
    model = ResUNet(in_channels=17, num_classes=2).to(device)
    criterion = CombinedFocalDiceLoss(focal_weight=0.35, dice_weight=0.65, alpha=0.50, gamma=2.0)
    
    print(f" Loading pretrained weights from {pretrained_model_path}....")
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    pretrained_dict = checkpoint['model_state_dict']
    
    # ------------ Loading the pretrained model ------------ #
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f" [INFO] Loaded {len(pretrained_dict)} matching layer dictionaries.")
    
    best_val_f1 = 0.0 
    
    def run_epoch(epoch, total_epochs, optimizer, scheduler, phase_name):
        nonlocal best_val_f1
        
        model.train()
        running_train_loss = 0.0
        for images, targets in train_loader:
            images, targets = image.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_train_loss = running_train_loss + loss.item()
            
        train_loss = running_train_loss / len(train_loader)
        
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = image.to(device), targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                running_val_loss = running_val_loss + loss.item()
                
                batch_metrics = compute_metrics(predictions, targets)
                total_tp += batch_metrics[0]
                total_fp += batch_metrics[1]
                total_fn += batch_metrics[2]
                
        val_loss = running_val_loss / len(val_loader)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)
        
        scheduler.step(1 - iou)
        
        print(
            f"[{phase_name}] Epoch [{epoch:02d}/{total_epochs}] "
            f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {iou:.4f} | Val F1: {f1:.4f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.6f}"            
        )
        
        # --- Save Checkpoint ---
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1
            }, save_path)
            print(f" => Saved new best model checkpoint! F1: {best_val_f1:.4f}")
        
                    
    # ========================================
    # PHASE 1: FREEZE ENCODER & TRAIN DECODER
    # ========================================
    print("\n" + "="*55)
    print("PHASE 1: Feature Extraction (Encoder Frozen) ")
    print("="*55)
    
    # Freeze Encoder leyer
    for name, param in model.named_parameters():
        if 'enc' in name or 'bottleneck' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # Decoder and output remain active
            
    optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode="min", patience=3, factor=0.5)
    
    for epoch in range(1, phase1_epochs + 1):
        run_epoch(epoch, phase1_epochs, optimizer_p1, scheduler_p1, "Phase 1")
        
    # ========================================
    # PHASE 2: UNFREEZE ALL AND FINE TUNING
    # ========================================
    
    print("\n" + "="*55)
    print(" PHASE 2: Full Fine-Tuning (All Layers Unfrozen) ")
    print("="*55)
    
    # Unfreeze all the layer
    
    for param in model.parameters():
        param.requires_grad = True
        
    # Much Lower learining rate
    optimizer_p2 = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode="min", patience=5, factor=0.5)
    
    for epoch in range(1, phase2_epochs + 1):
        run_epoch(epoch, phase2_epochs, optimizer_p2, scheduler_p2, "Phase 2")
        
    # ==========================================
    # FINAL PHASE: UNBIASED TEST EVALUATION
    # ==========================================
    print("\n" + "="*55)
    print(" EVALUATING BEST MODEL ON UNSEEN TEST SET ")
    print("="*55)
    
    # loading the best weights found during training
    best_checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    test_tp, test_fp, test_fn = 0, 0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            batch_metrics = compute_metrics(predictions, targets)
            test_tp += batch_metrics[0]
            test_fp += batch_metrics[1]
            test_fn += batch_metrics[2]

    test_iou = test_tp / (test_tp + test_fp + test_fn + 1e-6)
    test_f1 = 2 * test_tp / (2 * test_tp + test_fp + test_fn + 1e-6)
    test_precision = test_tp / (test_tp + test_fp + 1e-6)
    test_recall = test_tp / (test_tp + test_fn + 1e-6)

    print(f"Final Test Metrics -> IoU: {test_iou:.4f} | F1: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}\n")
    print(f"[SUCCESS] Transfer Learning Pipeline Complete. Model saved to: {save_path}")
    
    return model