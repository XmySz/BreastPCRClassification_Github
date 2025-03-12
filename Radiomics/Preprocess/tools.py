import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Union
from typing import List

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_nifti_shape(file_path: Union[str, Path]) -> Dict[str, Union[tuple, str]]:
    """
    获取单个NIfTI文件的形状信息

    Args:
        file_path: NIfTI文件的路径

    Returns:
        包含文件名和形状信息的字典

    Raises:
        Exception: 当文件读取失败时抛出异常
    """
    try:
        # 确保file_path是Path对象
        file_path = Path(file_path)

        # 验证文件存在性
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 验证文件扩展名
        if not str(file_path).endswith(('.nii', '.nii.gz')):
            raise ValueError(f"不支持的文件格式: {file_path}")

        # 读取图像
        img = sitk.ReadImage(str(file_path))

        # 获取形状信息
        shape = img.GetSize()
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()

        return {
            'filename': file_path.name,
            'shape': shape,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'dimensions': len(shape)
        }

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
        raise


def analyze_nifti_directory(
        directory_path: Union[str, Path],
        recursive: bool = False,
        pattern: str = "*.nii.gz"
) -> List[Dict[str, Union[tuple, str]]]:
    """
    分析目录中所有NIfTI文件的形状信息

    Args:
        directory_path: 要分析的目录路径
        recursive: 是否递归搜索子目录
        pattern: 文件匹配模式，默认为"*.nii.gz"

    Returns:
        包含所有文件形状信息的列表

    Raises:
        Exception: 当目录不存在或处理过程中出现错误时抛出异常
    """
    try:
        # 确保directory_path是Path对象
        directory_path = Path(directory_path)

        # 验证目录存在性
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        # 获取文件列表
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))

        if not files:
            logger.warning(f"在目录 {directory_path} 中未找到匹配的文件")
            return []

        # 分析所有文件
        results = []
        total_files = len(files)

        logger.info(f"开始分析目录: {directory_path}")
        logger.info(f"共找到 {total_files} 个文件待处理")

        for i, file_path in enumerate(files, 1):
            try:
                logger.info(f"处理第 {i}/{total_files} 个文件: {file_path.name}")
                result = get_nifti_shape(file_path)
                results.append(result)

            except Exception as e:
                logger.error(f"处理文件 {file_path.name} 时出错: {str(e)}")
                continue

        logger.info(f"分析完成，成功处理 {len(results)}/{total_files} 个文件")
        return results

    except Exception as e:
        logger.error(f"处理目录 {directory_path} 时发生错误: {str(e)}")
        raise


def print_shape_info(shape_info: Dict[str, Union[tuple, str]]):
    """
    格式化打印形状信息

    Args:
        shape_info: 包含形状信息的字典
    """
    print("\n" + "=" * 50)
    print(f"文件名: {shape_info['filename']}")
    print(f"形状: {shape_info['shape']}")
    print(f"体素间距: {shape_info['spacing']}")
    print(f"原点: {shape_info['origin']}")
    print(f"方向: {shape_info['direction']}")
    print(f"维度: {shape_info['dimensions']}")
    print("=" * 50 + "\n")


def CheckShapeMatchingByNib(imagesTr, labelsTr):
    images = sorted(glob.glob(imagesTr + '/*'))
    labels = sorted(glob.glob(labelsTr + '/*'))

    for i, (image, label) in enumerate(zip(images, labels), 1):
        nii_image = nib.load(image)
        nii_label = nib.load(label)

        img_shape = nii_image.header.get_data_shape()
        label_shape = nii_label.header.get_data_shape()

        print(f"{img_shape} {label_shape}")

        if img_shape != label_shape:
            print(os.path.basename(image), img_shape)

        # assert img_shape == label_shape, image


def CheckShapeMatchingByITK(imagesTr, labelsTr):
    images = sorted(glob.glob(imagesTr + '/*'))
    labels = sorted(glob.glob(labelsTr + '/*'))

    for i, (image, label) in enumerate(zip(images, labels), 1):
        img_sitk = sitk.ReadImage(image)
        label_sitk = sitk.ReadImage(label)

        img_shape = img_sitk.GetSize()
        label_shape = label_sitk.GetSize()

        print(image)
        print(f"{img_shape} {label_shape}")

        if img_shape != label_shape:
            print(os.path.basename(image), img_shape)

        # assert img_shape == label_shape, image


def calc_treatment_diff(pre_file, post_file, output_file):
    # 读取数据
    post_df = pd.read_excel(post_file)
    pre_df = pd.read_excel(pre_file)

    # 保存Patient列并计算差值
    patients = post_df['Patient'].copy()
    diff_df = post_df.drop('Patient', axis=1).subtract(pre_df.drop('Patient', axis=1))

    # 将Patient列插入到第一列位置
    diff_df.insert(0, 'Patient', patients)

    # 保存结果
    diff_df.to_excel(output_file, index=False)
    return diff_df


def organize_medical_data(src_root: Union[str, Path],
                          dst_root: Union[str, Path]) -> None:
    """
    整理医疗影像数据，将源目录中的文件按照标准化结构复制到目标目录

    参数:
        src_root: 包含患者子目录的源根目录
        dst_root: 用于存放整理后数据的目标根目录
    """
    # 将输入路径转换为Path对象，便于路径操作
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 定义目标目录结构
    dir_structure = {
        'PreTreatment': {  # 治疗前
            'Images': src_root / 'T0' / 'image.nii.gz',  # 影像文件
            'Labels': src_root / 'T0' / 'label.nii.gz'  # 标注文件
        },
        'PostTreatment': {  # 治疗后
            'Images': src_root / 'T2' / 'image.nii.gz',  # 影像文件
            'Labels': src_root / 'T2' / 'label.nii.gz'  # 标注文件
        }
    }

    # 创建目标目录结构
    for treatment in dir_structure:
        for subdir in ['Images', 'Labels']:
            os.makedirs(dst_root / treatment / subdir, exist_ok=True)

    # 处理每个患者的目录
    for patient_dir in sorted(src_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        # 按照目录结构复制文件
        try:
            for treatment, paths in dir_structure.items():
                for subdir, src_path in paths.items():
                    # 构建源文件和目标文件的完整路径
                    subdir_name = 'pretreatment' if treatment == 'PreTreatment' else 'treatment'
                    src_file = patient_dir / subdir_name / ('image.nii.gz' if subdir == 'Images' else 'label.nii.gz')
                    dst_file = dst_root / treatment / subdir / f"{patient_id}.nii.gz"
                    print(src_file, dst_file)
                    # 复制文件，保留元数据
                    shutil.copy2(src_file, dst_file)
        except (FileNotFoundError, PermissionError) as e:
            print(f"处理患者 {patient_id} 的文件时出错: {str(e)}")


def is_problematic_mask(nonzero_indices):
    """
    检查非零切片索引列表是否存在问题
    问题定义为:存在一个值远离主要的连续序列
    """
    if len(nonzero_indices) <= 1:
        return False, None

    # 计算相邻索引之间的差值
    gaps = np.diff(nonzero_indices)

    # 找出不连续的地方(差值大于1的位置)
    break_points = np.where(gaps > 1)[0]

    if len(break_points) == 0:
        # 所有索引都是连续的
        return False, None

    if len(break_points) > 1:
        # 有多个不连续点，不符合"一段连续+一个离群值"的模式
        return False, None

    # 找出哪段更短（更可能是离群值）
    first_segment = nonzero_indices[:break_points[0] + 1]
    second_segment = nonzero_indices[break_points[0] + 1:]

    if len(first_segment) == 1:
        return True, nonzero_indices[0]
    elif len(second_segment) == 1:
        return True, nonzero_indices[-1]

    return False, None


def process_mask_file(filepath):
    """处理单个mask文件,仅保留最长连续子序列"""
    # 读取nii.gz文件
    img = nib.load(filepath)
    data = img.get_fdata()

    # 获取非零切片的索引
    nonzero_slices = []
    for i in range(data.shape[2]):  # 假设第三维是切片维度
        if np.any(data[:, :, i] != 0):
            nonzero_slices.append(i)

    if not nonzero_slices:
        return

    # 找出最长连续子序列
    longest_seq = []
    current_seq = [nonzero_slices[0]]
    
    for i in range(1, len(nonzero_slices)):
        if nonzero_slices[i] == nonzero_slices[i-1] + 1:
            current_seq.append(nonzero_slices[i])
        else:
            if len(current_seq) > len(longest_seq):
                longest_seq = current_seq
            current_seq = [nonzero_slices[i]]
    
    if len(current_seq) > len(longest_seq):
        longest_seq = current_seq

    # 将不在最长连续子序列中的切片置零
    for i in nonzero_slices:
        if i not in longest_seq:
            data[:, :, i] = 0
            print(f"将切片 {i} 置零 (文件: {filepath})")

    # 保存修改后的文件
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, filepath)
    print(f"已保留最长连续序列: {longest_seq}")


def process_directory(root_dir):
    """处理目录下的所有mask文件"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "label.nii.gz":
                filepath = os.path.join(dirpath, filename)
                try:
                    # 从文件名中提取病例编号
                    case_id = int(os.path.basename(os.path.dirname(os.path.dirname(filepath))))
                    if case_id < 5731:
                        continue
                    print(filepath)
                    process_mask_file(filepath)
                except Exception as e:
                    print(f"处理文件 {filepath} 时出错: {str(e)}")
                    continue


if __name__ == "__main__":
    # organize_medical_data(r"F:\Data\HX\Dataset\center1\Orginal", r"F:\Data\HX\Dataset\center1\Standardized")
    calc_treatment_diff(r"F:\Data\BreastClassification\Experiment\Radiomics\center2\PreTreatment.xlsx",
                        r"F:\Data\BreastClassification\Experiment\Radiomics\center2\PostTreatment.xlsx",
                        r"F:\Data\BreastClassification\Experiment\Radiomics\center2\Difference.xlsx")