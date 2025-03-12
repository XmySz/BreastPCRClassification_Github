import os
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SaveImaged
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def complete_sequence(seq):
    """补全长度为3的序列到长度4"""
    parts = seq.split('->')
    if len(parts) == 3:
        all_timepoints = ['T0', 'T1', 'T2', 'T3']
        missing = [t for t in all_timepoints if t not in parts][0]
        parts.append(missing)
    return '->'.join(parts)


def load_and_preprocess_image(image_path):
    """加载并预处理3D医学图像
    
    参数:
        image_path: nii.gz格式图像路径
        
    返回:
        预处理后的3D图像数组,归一化到[0,1]范围
    """
    try:
        # 加载nii.gz格式图像
        img = nib.load(image_path)
        img_array = img.get_fdata(dtype=np.float32)

        # 归一化到[0,1]范围
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

        return img_array
    except Exception as e:
        print(f"加载图像 {image_path} 时出错: {e}")
        return None


def process_single_case(row, root_dir):
    """处理单个病例的数据
    
    参数:
        row: DataFrame的一行数据
        root_dir: 图像根目录
    """
    path = str(row['path'])
    combination = row['combination']
    label = row['label']

    # 如果序列长度为3则补全到4
    if len(combination.split('->')) == 3:
        full_sequence = complete_sequence(combination)
    else:
        full_sequence = combination

    timepoints = full_sequence.split('->')

    # 加载序列中的所有图像
    sequence_images = []
    valid_sequence = True

    for timepoint in timepoints:
        img_path = os.path.join(root_dir, str(path), timepoint, 'image.nii.gz')
        if not os.path.exists(img_path):
            print(f"缺失图像: {img_path}")
            # 使用空数组代替缺失图像
            img_array = np.zeros((128, 128, 32))
            sequence_images.append(img_array)
            continue

        img_array = load_and_preprocess_image(img_path)
        if img_array is None:
            valid_sequence = False
            break
        sequence_images.append(img_array)

    if valid_sequence:
        sequence_images = [img.astype(np.float32) for img in sequence_images]
        return {
            'patient_id': path,
            'timepoints': timepoints,
            'images': np.stack(sequence_images),
            'label': label
        }
    return None


def build_dataset(excel_path, root_dir, test_size=0.2, random_state=42, max_workers=4):
    """构建训练集和验证集数据集
    
    参数:
        excel_path: Excel文件路径
        root_dir: 图像根目录
        test_size: 验证集比例
        random_state: 随机种子
        max_workers: 最大并行工作进程数
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 初始化列表存储数据
    valid_data = []

    # 使用线程池并行处理数据
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_row = {
            executor.submit(process_single_case, row, root_dir): idx 
            for idx, row in df.iterrows()
        }
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="处理数据"):
            result = future.result()
            if result is not None:
                valid_data.append(result)

    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        valid_data, test_size=test_size, random_state=random_state,
        stratify=[d['label'] for d in valid_data]
    )
    
    # 创建保存目录
    save_dir = os.path.dirname(excel_path)
    save_path = os.path.join(save_dir, 'fine_tuning_dataset')
    train_dir = os.path.join(save_path, 'train')
    val_dir = os.path.join(save_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 并行保存训练集和验证集
    def save_single_data(data, split_dir):
        timepoint_str = '_'.join(data['timepoints'])
        filename = f"{data['patient_id']}_{timepoint_str}.npy"
        save_file = os.path.join(split_dir, filename)
        np.save(save_file, {
            'images': data['images'],
            'label': data['label']
        })
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 保存训练集
        train_futures = [
            executor.submit(save_single_data, data, train_dir)
            for data in train_data
        ]
        # 保存验证集
        val_futures = [
            executor.submit(save_single_data, data, val_dir)
            for data in val_data
        ]
        
        # 等待所有保存任务完成
        for future in tqdm(as_completed(train_futures + val_futures), 
                         total=len(train_futures) + len(val_futures),
                         desc="保存数据"):
            pass
    
    # 保存数据集信息
    with open(os.path.join(save_path, 'dataset_info.txt'), 'w') as f:
        f.write(f"总样本数: {len(valid_data)}\n")
        f.write(f"训练集样本数: {len(train_data)}\n")
        f.write(f"验证集样本数: {len(val_data)}\n")
        f.write(f"正样本数: {sum(d['label'] for d in valid_data)}\n")
        f.write(f"负样本数: {len(valid_data) - sum(d['label'] for d in valid_data)}\n")
    
    print(f"数据集统计信息:")
    print(f"总样本数: {len(valid_data)}")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"正样本数: {sum(d['label'] for d in valid_data)}")
    print(f"负样本数: {len(valid_data) - sum(d['label'] for d in valid_data)}")
    print(f"数据集已保存至: {save_path}")
    
    return save_path


if __name__ == "__main__":
    excel_path = r"F:\Data\HX\Dataset\ISPY2_SELECT\FineTuningDataset\微调数据集映射表.xlsx"
    root_dir = r"F:\Data\HX\Dataset\ISPY2_SELECT\Cropped_OnlyROI"      # 图像根目录
    build_dataset(excel_path, root_dir)