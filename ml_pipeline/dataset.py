import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional

class PlantVillageDataset(Dataset):
    """
    Custom Dataset for loading plant leaf images with Albumentations augmentation support.
    Expects directory structure: root_dir/class_name/image_filename.jpg
    """
    def __init__(self, root_dir: str, transform: Optional[A.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.samples = self._make_dataset(root_dir, self.class_to_idx)

    def _find_classes(self, dir_path: str) -> Tuple[List[str], Dict[str, int]]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
        classes = sorted([d.name for d in os.scandir(dir_path) if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir_path: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
        images = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for target_class in sorted(class_to_idx.keys()):
            class_dir = os.path.join(dir_path, target_class)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if os.path.splitext(fname)[1].lower() in valid_extensions:
                        path = os.path.join(root, fname)
                        images.append((path, class_to_idx[target_class]))
        return images

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        
        # Load via OpenCV (fastest decoding) - forces 3-channel BGR
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image at path: {path}")
            
        # Convert BGR to RGB for Albumentations & PyTorch models
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback default transform if none provided
            default_tf = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            image = default_tf(image=image)["image"]

        return image, target

def get_transforms(img_size: int = 224) -> Tuple[A.Compose, A.Compose]:
    """Returns training and validation albumentations transform pipelines."""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3), # Cutout regularization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform

def get_dataloaders(
    train_dir: str, 
    val_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4, 
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Initializes datasets and returns production-ready DataLoaders."""
    train_tf, val_tf = get_transforms(img_size)
    
    train_dataset = PlantVillageDataset(train_dir, transform=train_tf)
    val_dataset = PlantVillageDataset(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_dataset.classes