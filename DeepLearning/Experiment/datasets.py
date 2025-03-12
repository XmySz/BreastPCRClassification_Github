import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from monai.transforms import (
    Compose,
    RandAffined,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandRotated,
    ScaleIntensityd,
    EnsureTyped,
    NormalizeIntensityd
)


class BreastPCRDataset(Dataset):
    def __init__(self, data_path, train=True):
        """
        初始化数据集

        参数:
            data_path (str): 数据路径
            train (bool): 是否为训练集
        """
        from sklearn.model_selection import train_test_split

        self.data_path = data_path
        file_list = os.listdir(self.data_path)

        # 固定验证集比例和随机种子
        val_size = 0.2
        random_state = 42

        # 获取所有文件的标签
        labels = []
        for file in file_list:
            data = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            labels.append(data['label'])

        # 使用sklearn划分训练集和验证集
        train_files, val_files = train_test_split(
            file_list,
            test_size=val_size,
            random_state=random_state,
            stratify=labels  # 确保分割后的数据集保持原始标签分布
        )

        # 根据train标志位选择对应的文件列表
        self.file_list = train_files if train else val_files

        # 定义数据增强转换
        if train:
            self.transforms = Compose([
                EnsureTyped(keys=["images"]),
                NormalizeIntensityd(keys=["images"], nonzero=True),  # 执行z-score归一化
                # NormalizeIntensityd(keys=["images"], nonzero=True, subtrahend=244.46, divisor=383.74),  # 执行z-score归一化
                ScaleIntensityd(keys=["images"]),
                # RandAffined(
                #     keys=["images"],
                #     prob=0.10,
                #     rotate_range=(0.15, 0.15, 0.15),
                #     scale_range=(0.1, 0.1, 0.1),
                #     translate_range=(5, 5, 5),
                #     padding_mode="zeros"
                # ),
                # RandAdjustContrastd(
                #     keys=["images"],
                #     prob=0.15,
                #     gamma=(0.5, 2.0)
                # ),
                # RandRotated(
                #     keys=["images"],
                #     prob=0.2,
                #     range_x=15.0,
                #     range_y=15.0,
                #     range_z=15.0,
                #     padding_mode="zeros"
                # )
            ])
        else:
            self.transforms = Compose([
                EnsureTyped(keys=["images"]),
                NormalizeIntensityd(keys=["images"], nonzero=True),  # 执行z-score归一化
                # NormalizeIntensityd(keys=["images"], nonzero=True, subtrahend=244.46, divisor=383.74),  # 执行z-score归一化
                ScaleIntensityd(keys=["images"]),
                # RandAffined(
                #     keys=["images"],
                #     prob=0.10,
                #     rotate_range=(0.15, 0.15, 0.15),
                #     scale_range=(0.1, 0.1, 0.1),
                #     translate_range=(5, 5, 5),
                #     padding_mode="zeros"
                # ),
                # RandAdjustContrastd(
                #     keys=["images"],
                #     prob=0.15,
                #     gamma=(0.5, 2.0)
                # ),
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 获取npy文件路径
        file_path = os.path.join(self.data_path, self.file_list[idx])

        # 修改加载方式
        data = np.load(file_path, allow_pickle=True).item()  # 使用 numpy 加载

        # 构建数据字典
        data_dict = {
            'images': data['images'],
            'label': data['label']
        }

        # 应用数据增强
        data_dict = self.transforms(data_dict)

        return {
            'images': data_dict['images'].float(),
            'label': torch.tensor(data_dict['label']).long()
        }


if __name__ == "__main__":
    data_dir = r"F:\Data\HX\Dataset\ISPY2_SELECT\FineTuningDataset\Images"  # 数据集根目录
    dataset = BreastPCRDataset(data_dir)

    # 创建训练集和验证集
    train_dataset = BreastPCRDataset(data_dir, train=True)
    val_dataset = BreastPCRDataset(data_dir, train=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # 测试训练集
    print("\n开始测试训练集:")
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images']  # [B, 4, H, W, D]
        labels = batch['label']  # [B]

        print(f"\n训练集第 {batch_idx + 1} 个批次:")
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels.tolist()}")

        if batch_idx >= 2:  # 只测试前3个批次
            break

    # 测试验证集
    print("\n开始测试验证集:")
    for batch_idx, batch in enumerate(val_loader):
        images = batch['images']  # [B, 4, H, W, D]
        labels = batch['label']  # [B]

        print(f"\n验证集第 {batch_idx + 1} 个批次:")
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels.tolist()}")