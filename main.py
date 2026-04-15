import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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


import os, tempfile

def save_checkpoint(state, path, is_best=False):
    # atomic write — save to temp file first, then rename
    # if the job dies mid-write, the previous checkpoint survives
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix='.tmp') as f:
        torch.save(state, f.name)
        tmp_path = f.name
    os.replace(tmp_path, path)  # atomic on Linux
    
    if is_best:
        best_path = path.replace('.pth.tar', '_best.pth.tar')
        with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix='.tmp') as f:
            torch.save(state, f.name)
            tmp_path = f.name
        os.replace(tmp_path, best_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Run data approximation model')
    parser.add_argument('--run',
                        type=str)
    parser.add_argument('--resume',
                        action='store_true')
    parser.add_argument('--unfreeze',
                    action='store_true')

    cli_args = parser.parse_args()

    DATA_PATH = "/scratch/gssodhi/data_extract"
    MODEL_SAVE_PATH = f"/scratch/gssodhi/data_extract/checkpoint/{cli_args.run}.pth.tar"
    
    log_path = f"/home/gssodhi/comp_vis/Data_Extraction/data/{cli_args.run}.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HeatmapModel()

    model = model.to(device)
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

    
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
    cosine    = CosineAnnealingLR(optimizer, T_max=47, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])

    start_epoch = 0
    if cli_args.resume:
        loaded_state = torch.load(MODEL_SAVE_PATH, weights_only=False)
        start_epoch = loaded_state['epoch'] + 1
        model.load_state_dict(loaded_state['state_dict'])
        
        if cli_args.unfreeze:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': 1e-4}
            ])
            # fresh scheduler for fine-tuning phase
            scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
        else:
            optimizer.load_state_dict(loaded_state['optimizer'])
            scheduler.load_state_dict(loaded_state['scheduler'])

    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = test(model, val_loader, criterion, device)
        scheduler.step()
    
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
    
        save_checkpoint(state, MODEL_SAVE_PATH, is_best=(val_loss < best_val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss])
        
        print(f"epoch: {epoch+1}/{config.EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
