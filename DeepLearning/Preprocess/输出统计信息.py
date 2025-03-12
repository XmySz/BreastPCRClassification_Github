import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, List
import multiprocessing as mp
from tqdm import tqdm


def get_roi_statistics(label_data: np.ndarray) -> Dict:
    """
    计算单个样本的ROI统计信息

    Parameters:
        label_data (np.ndarray): 3D标签数组

    Returns:
        Dict: 包含ROI统计信息的字典
    """
    # 找到所有非零像素的坐标
    roi_coords = np.where(label_data > 0)

    if len(roi_coords[0]) == 0:
        return {
            '开始切片': None,
            '结束切片': None,
            'ROI层数': 0,
            '最小正方体': None,
            '最小正方体形状': None,
            'ROI体积': 0
        }

    # 计算包含ROI的切片范围
    start_slice = np.min(roi_coords[2])
    end_slice = np.max(roi_coords[2])
    roi_slice_count = end_slice - start_slice + 1

    # 计算最小包围盒
    min_coords = np.min(roi_coords, axis=1)
    max_coords = np.max(roi_coords, axis=1)
    dimensions = max_coords - min_coords + 1
    min_bounding_cube = int(np.max(dimensions))
    min_bounding_shape = tuple(int(d) for d in dimensions)

    # 计算ROI体积（非零像素数量）
    roi_volume = np.sum(label_data > 0)

    return {
        '开始切片': int(start_slice),
        '结束切片': int(end_slice),
        'ROI层数': int(roi_slice_count),
        '最小正方体': min_bounding_cube,
        '最小正方体形状': str(min_bounding_shape),
        'ROI体积': int(roi_volume)
    }


def process_single_sample(sample_info: Tuple[str, str, str]) -> Dict:
    """
    处理单个样本的图像和标签数据

    Parameters:
        sample_info (Tuple[str, str, str]): (患者ID, 时间点, 文件路径)的元组

    Returns:
        Dict: 包含样本统计信息的字典
    """
    patient_id, timepoint, base_path = sample_info
    
    img_file = Path(base_path) / "image.nii.gz"
    label_file = Path(base_path) / "label.nii.gz"

    try:
        # 加载图像和标签数据
        img_nib = nib.load(str(img_file))
        label_nib = nib.load(str(label_file))

        img_data = img_nib.get_fdata()
        label_data = label_nib.get_fdata()

        # 获取ROI统计信息
        roi_stats = get_roi_statistics(label_data)

        # 收集样本信息
        sample_info = {
            '患者ID': patient_id,
            '时间点': timepoint,
            '图像尺寸': f"{img_data.shape}",
            '图像间距': f"{img_nib.header.get_zooms()}",
            '图像强度范围': f"({np.min(img_data):.2f}, {np.max(img_data):.2f})",
            '图像强度均值': f"{np.mean(img_data):.2f}",
            '图像强度标准差': f"{np.std(img_data):.2f}",
            **roi_stats
        }

        return sample_info

    except Exception as e:
        print(f"处理样本 {patient_id}-{timepoint} 时发生错误: {str(e)}")
        return None


def analyze_breast_mri_dataset(base_path: str, num_processes: int = None) -> pd.DataFrame:
    """
    并行分析3D乳腺癌MRI数据集，统计ROI信息和图像特征

    Parameters:
        base_path (str): 数据集根目录路径
        num_processes (int, optional): 并行处理的进程数，默认为CPU核心数

    Returns:
        pd.DataFrame: 包含每个样本统计信息的DataFrame
    """
    base_path = Path(base_path)
    
    # 获取所有患者和时间点
    sample_list = []
    for patient_dir in base_path.iterdir():
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            for timepoint_dir in patient_dir.iterdir():
                if timepoint_dir.is_dir():
                    timepoint = timepoint_dir.name
                    sample_list.append((patient_id, timepoint, str(timepoint_dir)))


    # 设置进程数
    if num_processes is None:
        num_processes = mp.cpu_count()

    # 创建进程池
    pool = mp.Pool(processes=num_processes)

    try:
        # 使用tqdm显示进度条
        results = list(tqdm(
            pool.imap(process_single_sample, sample_list),
            total=len(sample_list),
            desc="处理样本"
        ))
    finally:
        pool.close()
        pool.join()

    # 过滤掉处理失败的样本
    dataset_stats = [r for r in results if r is not None]

    # 创建DataFrame并设置列顺序
    columns = [
        '患者ID', '时间点', '图像尺寸', '图像间距',
        '开始切片', '结束切片', 'ROI层数',
        '最小正方体', '最小正方体形状', 'ROI体积',
        '图像强度范围', '图像强度均值', '图像强度标准差'
    ]

    df = pd.DataFrame(dataset_stats)
    df = df[columns]

    return df


if __name__ == "__main__":
    # 设置数据路径
    base_dir = r"F:\Data\BreastClassification\Dataset\center1\Original"
    num_processes = 6

    try:
        # 分析数据集
        print(f"\n使用 {num_processes or mp.cpu_count()} 个进程进行并行处理...")
        stats_df = analyze_breast_mri_dataset(base_dir, num_processes)

        # 打印结果
        print("\n数据集统计信息:")
        print(stats_df)

        output_file = Path(base_dir).parent / "Dataset_Statistics.xlsx"
        stats_df.to_excel(str(output_file), index=False)
        print(f"\n统计结果已保存到 {output_file}")

    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")