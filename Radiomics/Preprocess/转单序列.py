import os
from pathlib import Path
from typing import Union, List, Tuple, Dict
import nibabel as nib
import numpy as np
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def reduce_nifti_dimensions(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = None,
        recursive: bool = False,
        keep_original: bool = True
) -> List[Dict]:
    """
    遍历目录中的NIfTI文件，将5维数据降为3维（保留第一个序列）

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则在原目录创建processed子目录
        recursive: 是否递归搜索子目录
        keep_original: 是否保留原始文件

    Returns:
        处理结果列表，包含文件信息和处理状态
    """
    try:
        # 转换路径为Path对象
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir / 'processed'
        else:
            output_dir = Path(output_dir)

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有nii.gz文件
        if recursive:
            files = list(input_dir.rglob('*.nii.gz'))
        else:
            files = list(input_dir.glob('*.nii.gz'))

        if not files:
            logger.warning(f"在目录 {input_dir} 中未找到.nii.gz文件")
            return []

        logger.info(f"找到 {len(files)} 个.nii.gz文件")
        results = []

        # 使用tqdm显示进度
        for file_path in tqdm(files, desc="处理文件"):
            try:
                # 加载nifti文件
                img = nib.load(str(file_path))
                original_shape = img.shape

                result_dict = {
                    'filename': file_path.name,
                    'original_shape': original_shape,
                    'status': 'skipped',
                    'new_shape': None,
                    'error': None
                }

                # 如果是5维数据
                if len(original_shape) == 5:
                    logger.info(f"处理5维文件: {file_path.name}, 原始形状: {original_shape}")

                    # 获取图像数据
                    data = img.get_fdata()

                    # 提取第一个序列
                    reduced_data = data[..., 0, 0]  # 假设最后两个维度是我们要去除的

                    # 创建新的nifti图像
                    # 获取原始图像的affine矩阵
                    affine = img.affine

                    # 创建新的头信息
                    new_header = img.header.copy()
                    # 更新头信息中的维度
                    new_header.set_data_shape(reduced_data.shape)

                    # 创建新的nifti图像
                    reduced_img = nib.Nifti1Image(reduced_data, affine, new_header)

                    # 构建输出文件路径
                    output_path = output_dir / file_path.name

                    # 保存处理后的图像
                    nib.save(reduced_img, str(output_path))

                    # 如果不保留原始文件且输出目录不同于输入目录
                    if not keep_original and output_dir != input_dir:
                        file_path.unlink()  # 删除原始文件

                    result_dict.update({
                        'status': 'processed',
                        'new_shape': reduced_data.shape
                    })

                    logger.info(f"文件 {file_path.name} 处理完成，新形状: {reduced_data.shape}")
                else:
                    logger.info(f"跳过非5维文件: {file_path.name}, 形状: {original_shape}")

                results.append(result_dict)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"处理文件 {file_path.name} 时出错: {error_msg}")
                results.append({
                    'filename': file_path.name,
                    'original_shape': None,
                    'status': 'error',
                    'new_shape': None,
                    'error': error_msg
                })
                continue

        logger.info(f"处理完成，共处理 {len([r for r in results if r['status'] == 'processed'])} 个5维文件")
        return results

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise


def print_processing_results(results: List[Dict]):
    """
    打印处理结果

    Args:
        results: 处理结果列表
    """
    print("\n" + "=" * 50)
    print("处理结果汇总:")
    print("=" * 50)

    # 统计数据
    processed = len([r for r in results if r['status'] == 'processed'])
    skipped = len([r for r in results if r['status'] == 'skipped'])
    errors = len([r for r in results if r['status'] == 'error'])

    print(f"\n总文件数: {len(results)}")
    print(f"成功处理: {processed}")
    print(f"跳过文件: {skipped}")
    print(f"处理失败: {errors}")

    print("\n详细信息:")
    print("-" * 50)

    for result in results:
        print(f"\n文件名: {result['filename']}")
        print(f"处理状态: {result['status']}")
        print(f"原始形状: {result['original_shape']}")

        if result['status'] == 'processed':
            print(f"处理后形状: {result['new_shape']}")
        elif result['status'] == 'error':
            print(f"错误信息: {result['error']}")

    print("\n" + "=" * 50)


def analyze_directory_shapes(input_dir: Union[str, Path], recursive: bool = False) -> Dict:
    """
    分析目录中所有nii.gz文件的形状分布

    Args:
        input_dir: 输入目录路径
        recursive: 是否递归搜索子目录

    Returns:
        形状分布统计信息
    """
    input_dir = Path(input_dir)
    shape_stats = {}

    # 获取文件列表
    pattern = '**/*.nii.gz' if recursive else '*.nii.gz'
    files = list(input_dir.glob(pattern))

    logger.info(f"开始分析目录: {input_dir}")
    logger.info(f"找到 {len(files)} 个.nii.gz文件")

    for file_path in tqdm(files, desc="分析文件形状"):
        try:
            img = nib.load(str(file_path))
            shape = img.shape
            shape_str = str(shape)

            if shape_str not in shape_stats:
                shape_stats[shape_str] = []
            shape_stats[shape_str].append(file_path.name)

        except Exception as e:
            logger.error(f"分析文件 {file_path.name} 时出错: {str(e)}")
            continue

    return shape_stats


# 使用示例
if __name__ == "__main__":
    try:
        # 设置输入输出路径
        input_directory = r"F:\Data\HX\Dataset\center1\Standardized\PostTreatment\Images"
        output_directory = r"F:\Data\HX\Dataset\center1\Standardized\PostTreatment\Images_1"  # 可选

        # 首先分析目录中的形状分布
        print("\n分析目录中的文件形状分布...")
        shape_stats = analyze_directory_shapes(input_directory, recursive=True)

        print("\n形状分布统计:")
        for shape, files in shape_stats.items():
            print(f"\n形状 {shape}:")
            print(f"文件数量: {len(files)}")
            print(f"示例文件: {files[0]}")

        # 询问用户是否继续处理
        response = input("\n是否继续处理5维文件？(y/n): ")

        if response.lower() == 'y':
            # 处理文件
            results = reduce_nifti_dimensions(
                input_directory,
                output_directory,
                recursive=True,
                keep_original=True
            )

            # 打印结果
            print_processing_results(results)
        else:
            print("操作已取消")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")