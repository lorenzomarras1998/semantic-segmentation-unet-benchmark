import os
import torch
from torch.utils.data import DataLoader, Subset
import torch_directml
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset
from metrics import BCEDiceLoss, dice_score

def train_transfer_learning():
    device = torch_directml.device()
    print(f"Device: {torch_directml.device_name(0)}")

    # 1. Dataset & Split (80/20)
    dataset_raw = SegmentationDataset("data/images", "data/masks", augment=False)
    n_val = int(len(dataset_raw) * 0.2)
    n_train = len(dataset_raw) - n_val

    train_ds_full = SegmentationDataset("data/images", "data/masks", augment=True)
    val_ds_full = SegmentationDataset("data/images", "data/masks", augment=False)

    train_set = Subset(train_ds_full, range(0, n_train))
    val_set = Subset(val_ds_full, range(n_train, len(dataset_raw)))

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    # 2. U-Net con Backbone ResNet34 pre-addestrato su ImageNet
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # 3. Loss ponderata & Optimizer (learning rate leggermente più basso per fine-tuning)
    criterion = BCEDiceLoss(alpha=0.6, beta=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_dice = 0.0
    epochs = 30  # Con i pesi pre-allenati bastano molte meno epoche

    print("Inizio addestramento U-Net (ResNet34 backbone)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(logits, masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validazione
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_dice += dice_score(logits, masks)

        val_dice /= len(val_loader)

        print(f"Epoca [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/unet_resnet34_pesi.pth")
            print(f"  -> Record salvato: {best_val_dice:.4f}")

    print(f"\nFine addestramento! Miglior Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    train_transfer_learning()