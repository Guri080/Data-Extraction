import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
from torch.optim import lr_scheduler

import os
import csv

from UNet import HeatmapModel
from dataloader import ScatterPointDataset
import training_config as config
from loss import AdaptiveWingLoss

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for img_mask in tqdm(loader, desc='train'):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        optimizer.zero_grad()

        y_pred = model(img)
        loss = criterion(y_pred, mask)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    total_loss = running_loss / len(loader)

    return total_loss

def test(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for img_mask in tqdm(loader, desc='val'):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
    
            y_pred = model(img)
            loss = criterion(y_pred, mask)
    
            running_loss += loss.item()
    
    total_loss = running_loss / len(loader)

    return total_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Run data approximation model')
    parser.add_argument('--run',
                        type=str)

    cli_args = parser.parse_args()

    DATA_PATH = "/scratch/gssodhi/data_extract"
    MODEL_SAVE_PATH = f"/scratch/gssodhi/data_extract/checkpoint/{cli_args.run}.pth.tar"
    
    log_path = f"/home/gssodhi/comp_vis/Data_Extraction/data/{cli_args.run}.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HeatmapModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = AdaptiveWingLoss()

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

    with open(log_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss'])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ## Training step
    for epoch in range(config.EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        scheduler.step()
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict()
        }

        torch.save(state, MODEL_SAVE_PATH)
        val_loss = test(model, val_loader, criterion, device)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss])
        
        print(f"epoch: {epoch+1}/{config.EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
