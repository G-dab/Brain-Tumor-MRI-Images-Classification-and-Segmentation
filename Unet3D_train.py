import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from Unet3D_dataset import MRIDataset
from Unet3D_model import UNet3D, Loss, dice

# Dataset and DataLoader
train_dataset = MRIDataset(root_dir='./dataset_segmentation/train', mode="train")
val_dataset = MRIDataset(root_dir='./dataset_segmentation/val', mode="test")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model, Loss, Optimizer
model = UNet3D(num_channels=1, feat_channels=[16, 32, 64, 128, 256]).to(device)
criterion = Loss()
optimizer = Adam(model.parameters(), lr=3e-4)

# Training
num_epochs = 2
train_loss_list = []
val_loss_list = []
scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

for epoch in range(num_epochs):
    train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)
    model.train()
    train_loss = []
    for data in train_loader_iter:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loader_iter.set_postfix({'Train Loss': loss.item(), 'Dice': dice(outputs, targets).item()})
        train_loss.append(loss.item())
    if epoch >= 4 and epoch <= 20:
        scheduler.step()

    model.eval()
    val_loss_total = 0.0
    val_dices = []
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            val_dice = dice(outputs, targets)
            val_loss_total += val_loss.item()
            val_dices.append(val_dice.item())

    avg_val_loss = val_loss_total / len(val_loader)
    avg_val_dice = np.mean(np.array(val_dices))
    std_val_dice = np.std(np.array(val_dices))
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {round(sum(train_loss) / len(train_loss), 5)}, '
          f'Val Loss: {round(avg_val_loss, 5)}, Dice on Val: {round(avg_val_dice, 5)}, '
          f'Dice Std: {round(std_val_dice, 5)}')

    train_loss_list.append(sum(train_loss) / len(train_loss))
    val_loss_list.append(val_loss_total)

    if avg_val_dice > 0.8:
        torch.save(model.state_dict(), './model_epoch.pth')

torch.save(model.state_dict(), './model.pth')