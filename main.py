import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import os

from UNet import HeatmapModel
from dataloader import ScatterPointDataset
import training_config as config

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for img_mask in tqdm(loader, desc='train'):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        optimizer.zero_grad()

        y_pred = model(img)
        loss = criterion(mask, y_pred)
        running_loss += loss.item()

        loss.backwards()
        optimizer.step()
    
    total_loss = running_loss / len(loader)

    return total_loss

def test(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0

    for img_mask in tqdm(loader, desc='val'):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = model(img)
        loss = criterion(mask, y_pred)

        running_loss += loss.item()
    
    total_loss = running_loss / len(loader)

    return total_loss

if __name__ == '__main__':
    DATA_PATH = "/scratch/gssodhi/data_extract"
    MODEL_SAVE_PATH = "/content/drive/MyDrive/uygar/unet-segmentation/models/unet.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HeatmapModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = ScatterPointDataset(os.path.join(DATA_PATH, 'train/images'),
                                        os.path.join(DATA_PATH, 'train/heatmaps') 
    )

    val_dataset = ScatterPointDataset(os.path.join(DATA_PATH, 'val/images'),
                                      os.path.join(DATA_PATH, 'val/heatmaps') 
    )
    
    train_loader = DataLoader(train_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4,
                            num_workers=config.NUM_WORKERS
                        )

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4,
                            num_workers=config.NUM_WORKERS
                        )


    for epoch in EPOCHS:
        train_loss = train(model, train_loader, optimizer, criterion, device)

        val_loss = test(model, val_loader, optimizer, device)

        print(f"epoch: {epoch} | train loss: {train_loss:4f} | val loss: {val_loss:.4f}")



