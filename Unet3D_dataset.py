import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchio as tio
from monai.transforms import Compose, RandAffine, EnsureChannelFirst, RandFlip, RandRotate
import monai.transforms as mt

transforms_ex1 = mt.Compose([
        # 随机翻转
        mt.RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
        mt.RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),

        # 随机旋转
        mt.RandRotated(keys=["image", "seg"], prob=0.5, range_x=(-15, 15), range_y=(-15, 15), range_z=(-15, 15), mode="nearest", padding_mode="zeros"),

        # 随机缩放
        mt.RandZoomd(keys=["image", "seg"], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode="nearest"),

        # 添加噪声
        mt.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),

        # 随机亮度调整
        mt.RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.1),

        # 随机裁剪
        mt.RandSpatialCropd(keys=["image", "seg"], roi_size=(96, 96, 96), random_size=False),

        # 确保数据形状一致
        mt.EnsureTyped(keys=["image", "seg"], dtype="float32")
    ])

class MRIDataset(Dataset):
    def __init__(self, root_dir, mode="test", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.file_list = self.load_file()

    def load_file(self):
        file_list = []
        for file_name in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file_name)
            if os.path.isfile(file_path) and file_name.endswith("_fla.nii.gz"):
                seg_file_name = file_name.replace("_fla.nii.gz", "_seg.nii.gz")
                seg_file_path = os.path.join(self.root_dir, seg_file_name)
                if os.path.exists(seg_file_path):
                    file_list.append((file_path, seg_file_path))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, seg_path = self.file_list[idx]

        # 加载NIfTI文件
        img_nifti = nib.load(img_path)
        seg_nifti = nib.load(seg_path)
        img_data = img_nifti.get_fdata()
        seg_data = seg_nifti.get_fdata()

        # 归一化（仅针对非零区域）
        non_zero_mask = img_data != 0
        mean = np.mean(img_data[non_zero_mask])
        std = np.std(img_data[non_zero_mask])
        img_data[non_zero_mask] = (img_data[non_zero_mask] - mean) / std

        # 转换为PyTorch张量并添加通道维度
        img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)  # Shape: [1, H, W, D]
        seg_tensor = torch.from_numpy(seg_data).long().unsqueeze(0)   # Shape: [1, H, W, D]

        # 应用数据增强（需确保图像和分割同步变换）
        if self.transform:
            data_dict = {
                "image": img_tensor,
                "seg": seg_tensor
            }
            data_dict = self.transform(data_dict)
            img_tensor = data_dict["image"]
            seg_tensor = data_dict["seg"]

        # 调整维度顺序为 [Channel, Depth, Height, Width]
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        seg_tensor = seg_tensor.permute(0, 3, 1, 2)

        return img_tensor, seg_tensor
    

if __name__ == '__init__':
    # Dataset and DataLoader
    train_dataset_path = 'train'
    val_dataset_path = 'val'

    train_dataset = MRIDataset(root_dir=train_dataset_path, 
                            mode="train",
                            transform=transforms_ex1)
    val_dataset = MRIDataset(root_dir=val_dataset_path, 
                            mode="test")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f'训练集数量: {len(train_dataset)}')
    print(f'验证集数量: {len(val_dataset)}')