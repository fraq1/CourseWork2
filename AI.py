import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {DEVICE}")

DATASET_PATH = r"C:\Users\-PC\Downloads\GapDetectionDataset"

IMAGE_SIZE = 256
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 30


TRAIN_IMAGES = os.path.join(DATASET_PATH, "images", "Train")
TRAIN_MASKS = os.path.join(DATASET_PATH, "masks", "Train")

VAL_IMAGES = os.path.join(DATASET_PATH, "images", "Val")
VAL_MASKS = os.path.join(DATASET_PATH, "masks", "Val")

TEST_IMAGES = os.path.join(DATASET_PATH, "images", "Test")
TEST_MASKS = os.path.join(DATASET_PATH, "masks", "Test")


class ShelfDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        image = cv2.imread(os.path.join(self.image_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0)
        return image, mask


train_transform = A.Compose([
    A.Resize(256, 256),

    A.HorizontalFlip(p=0.5),

    A.RandomBrightnessContrast(p=0.6),
    A.RandomGamma(gamma_limit=(70, 140), p=0.3),
    A.CLAHE(p=0.3),

    A.HueSaturationValue(p=0.3),
    A.GaussianBlur(p=0.2),

    A.Rotate(limit=10, p=0.5),

    A.Lambda(
        image=lambda x, **kwargs: (
            cv2.convertScaleAbs(
                x,
                alpha=1.6 if random.random() < 0.3 else 1.0,
                beta=80 if random.random() < 0.3 else 0
            )
        )
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


train_dataset = ShelfDataset(TRAIN_IMAGES, TRAIN_MASKS, train_transform)
val_dataset   = ShelfDataset(VAL_IMAGES, VAL_MASKS, val_transform)
test_dataset  = ShelfDataset(TEST_IMAGES, TEST_MASKS, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)


dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def iou(preds, masks):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    inter = (preds * masks).sum()
    union = preds.sum() + masks.sum() - inter

    return (inter / (union + 1e-6)).item()

def f1(preds, masks):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    tp = (preds * masks).sum()
    prec = tp / (preds.sum() + 1e-6)
    rec = tp / (masks.sum() + 1e-6)

    return (2 * prec * rec / (prec + rec + 1e-6)).item()


def train_fn(loader):
    model.train()
    total = 0

    for img, mask in tqdm(loader, desc="train"):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        pred = model(img)

        loss = dice_loss(pred, mask) + bce_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


def eval_fn(loader):
    model.eval()
    loss_total = 0
    iou_total = 0
    f1_total = 0

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(img)

            loss = dice_loss(pred, mask) + bce_loss(pred, mask)

            loss_total += loss.item()
            iou_total += iou(pred, mask)
            f1_total += f1(pred, mask)

    n = len(loader)
    return loss_total/n, iou_total/n, f1_total/n


best = 0

for epoch in range(EPOCHS):

    print(f"\nEPOCH {epoch+1}/{EPOCHS}")

    tr_loss = train_fn(train_loader)
    val_loss, val_iou, val_f1 = eval_fn(val_loader)

    print(f"train loss: {tr_loss:.4f}")
    print(f"val loss:   {val_loss:.4f}")
    print(f"val IoU:    {val_iou:.4f}")
    print(f"val F1:     {val_f1:.4f}")

    if val_iou > best:
        best = val_iou
        torch.save(model.state_dict(), "best_model.pth")
        print("MODEL SAVED")

print("\nTESTING")

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

test_loss, test_iou, test_f1 = eval_fn(test_loader)

print(f"test loss: {test_loss:.4f}")
print(f"test IoU:  {test_iou:.4f}")
print(f"test F1:   {test_f1:.4f}")