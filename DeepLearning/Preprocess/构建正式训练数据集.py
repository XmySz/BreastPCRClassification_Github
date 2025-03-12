import os
import nibabel as nib
import numpy as np
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SaveImaged
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def complete_sequence(seq):
    """补全长度为3的序列到长度4,并保持有序"""
    parts = seq.split('->')
    if len(parts) == 3:
        all_timepoints = ['T0', 'T1', 'T2', 'T3']
        missing = [t for t in all_timepoints if t not in parts][0]
        # 找到缺失时间点的正确位置
        missing_idx = int(missing[1])
        parts.insert(missing_idx, missing)
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
    original_dirs = row['original_dirs']
    label = row['label']

    # 初始化4个时间点的图像列表
    sequence_images = [np.zeros((128, 128, 64), dtype=np.float32) for _ in range(4)]
    timepoints = ['T0', 'T1', 'T2', 'T3']  # 固定的4个时间点

    # 获取实际的时间点序列
    actual_timepoints = original_dirs.split('->')

    # 加载存在的图像
    for tp in actual_timepoints:
        img_path = os.path.join(root_dir, str(path), tp, 'image.nii.gz')
        if os.path.exists(img_path):
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                # 将图像放入对应位置
                idx = int(tp[1])  # 从时间点名称获取索引(T0->0, T1->1等)
                sequence_images[idx] = img_array.astype(np.float32)
            else:
                print(f"处理图像失败: {img_path}")
        else:
            print(f"缺失图像: {img_path}")

    return {
        'patient_id': path,
        'timepoints': timepoints,
        'images': np.stack(sequence_images),  # 保证形状为(4, 128, 128, 64)
        'label': label
    }


def build_dataset(excel_path, root_dir, save_dir, max_workers=4):
    """构建数据集

    参数:
        excel_path: Excel文件路径
        root_dir: 图像根目录
        save_dir: 保存目录
        max_workers: 最大并行工作进程数
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name='训练集映射表')

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

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 并行保存数据
    def save_single_data(data):
        timepoint_str = '_'.join(data['timepoints'])
        filename = f"{data['patient_id']}_{timepoint_str}.npy"
        save_file = os.path.join(save_dir, filename)
        np.save(save_file, {
            'images': data['images'],
            'label': data['label']
        })

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 保存所有数据
        futures = [
            executor.submit(save_single_data, data)
            for data in valid_data
        ]

        # 等待所有保存任务完成
        for future in tqdm(as_completed(futures),
                         total=len(futures),
                         desc="保存数据"):
            pass

    return save_dir


if __name__ == "__main__":
    excel_path = r"/media/ke/SSD_2T_2/HX/Dataset/center2/center2.xlsx"
    root_dir = r"/media/ke/SSD_2T_2/HX/Dataset/center2/Cropped_OnlyROI_3"
    save_dir = r'/media/ke/SSD_2T_2/HX/Dataset/center2'
    build_dataset(excel_path, root_dir, save_dir)