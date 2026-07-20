import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders
from model import CropDiseaseClassifier, export_to_torchscript, export_to_onnx

class EarlyStopping:
    """Terminates training if validation loss ceases to improve over a defined patience window."""
    def __init__(self, patience: int = 7, min_delta: float = 0.0005, checkpoint_path: str = "best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] Patience counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"[EarlyStopping] Validation loss decreased to {val_loss:.4f}. Saved checkpoint.")

def train_one_epoch(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to stabilize transformer backbones
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += images.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})
        
    return running_loss / total, correct / total

@torch.inference_mode()
def validate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    
    for images, targets in tqdm(dataloader, desc="Validating", leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += images.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    return running_loss / total, correct / total, np.array(all_targets), np.array(all_preds)

def generate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, 
    class_names: list, save_path: str = "confusion_matrix.png"
) -> None:
    """Generates and saves a publication-grade confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=class_names, yticklabels=class_names, cbar=False
    )
    plt.title("Crop Disease Classification - Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Edge-Optimized Crop Disease Classifier")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset root folder")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation dataset root folder")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--img_size", type=int, default=224, help="Input resolution")
    args = parser.parse_args()

    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[System] Initializing training pipeline on device: {device}")

    # Data ingestion
    train_loader, val_loader, class_names = get_dataloaders(
        args.train_dir, args.val_dir, batch_size=args.batch_size, img_size=args.img_size
    )
    num_classes = len(class_names)
    print(f"[Dataset] Recognized {num_classes} distinct disease classes.")

    # Model compilation
    model = CropDiseaseClassifier(num_classes=num_classes, model_name=args.model_name).to(device)
    
    # Loss, Optimizer, and Scheduling
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Prevents overconfidence on noisy leaf backgrounds
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=args.patience, checkpoint_path="best_model.pth")

    print("[Training] Beginning epoch execution loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("[System] Early stopping triggered. Terminating training loop.")
            break

    # Load optimal weights for evaluation and export
    print("\n[System] Restoring best checkpoint for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    
    # Final validation evaluation
    val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
    print(f"\n[Final Results] Best Validation Accuracy: {val_acc*100:.2f}% (Loss: {val_loss:.4f})")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    
    # Generate artifacts
    generate_confusion_matrix(y_true, y_pred, class_names, "crop_disease_confusion_matrix.png")
    export_to_torchscript(model, f"{args.model_name}_edge.pt", device, args.img_size)
    export_to_onnx(model, f"{args.model_name}_edge.onnx", device, args.img_size)
    print("[Success] All training tasks and model exports completed.")

if __name__ == "__main__":
    main()