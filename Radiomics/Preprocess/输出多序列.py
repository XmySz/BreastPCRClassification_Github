import os
from pathlib import Path
from typing import Union, List, Dict
import nibabel as nib
import numpy as np
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_sequences(
        input_dir: Union[str, Path],
        output_base_dir: Union[str, Path] = None,
        recursive: bool = False,
        keep_original: bool = True
) -> List[Dict]:
    """
    将5维NIfTI文件的每个时期序列分别保存到对应的子文件夹中

    Args:
        input_dir: 输入目录路径
        output_base_dir: 输出基础目录路径，如果为None则在原目录创建processed子目录
        recursive: 是否递归搜索子目录
        keep_original: 是否保留原始文件

    Returns:
        处理结果列表
    """
    try:
        # 转换路径为Path对象
        input_dir = Path(input_dir)
        if output_base_dir is None:
            output_base_dir = input_dir / 'processed'
        else:
            output_base_dir = Path(output_base_dir)

        # 确保输出基础目录存在
        output_base_dir.mkdir(parents=True, exist_ok=True)

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
                    'sequences_saved': [],
                    'error': None
                }

                # 如果是5维数据
                if len(original_shape) == 5:
                    logger.info(f"处理5维文件: {file_path.name}, 原始形状: {original_shape}")

                    # 获取图像数据
                    data = img.get_fdata()

                    # 创建以文件名（不包含扩展名）命名的子目录
                    base_name = file_path.stem
                    if base_name.endswith('.nii'):  # 处理.nii.gz情况
                        base_name = base_name[:-4]
                    output_dir = output_base_dir / base_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # 获取序列数量（第4维）
                    num_sequences = original_shape[3]
                    sequences_saved = []

                    # 遍历每个序列
                    for seq_idx in range(num_sequences):
                        # 提取当前序列的所有时间点
                        sequence_data = data[..., seq_idx, :]

                        # 遍历每个时间点
                        for time_idx in range(original_shape[4]):
                            # 提取当前时间点的3D数据
                            time_data = sequence_data[..., time_idx]

                            # 创建文件名：原始文件名_序列号_时间点号.nii.gz
                            output_name = f"{base_name}_s{seq_idx + 1}_t{time_idx + 1}.nii.gz"
                            output_path = output_dir / output_name

                            # 创建新的nifti图像
                            affine = img.affine
                            new_header = img.header.copy()
                            new_header.set_data_shape(time_data.shape)

                            reduced_img = nib.Nifti1Image(time_data, affine, new_header)

                            # 保存处理后的图像
                            nib.save(reduced_img, str(output_path))
                            sequences_saved.append(output_name)

                    # 如果不保留原始文件且输出目录不同于输入目录
                    if not keep_original and output_base_dir != input_dir:
                        file_path.unlink()  # 删除原始文件

                    result_dict.update({
                        'status': 'processed',
                        'sequences_saved': sequences_saved
                    })

                    logger.info(f"文件 {file_path.name} 处理完成，保存了 {len(sequences_saved)} 个序列")
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
                    'sequences_saved': [],
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
            print(f"保存的序列数: {len(result['sequences_saved'])}")
            print("保存的序列示例:")
            for i, seq in enumerate(result['sequences_saved'][:3]):  # 只显示前3个序列
                print(f"  - {seq}")
            if len(result['sequences_saved']) > 3:
                print("  ...")
        elif result['status'] == 'error':
            print(f"错误信息: {result['error']}")

    print("\n" + "=" * 50)


# 使用示例
if __name__ == "__main__":
    try:
        # 设置输入输出路径
        input_directory = r"F:\Data\HX\Dataset\center1\Standardized\PostTreatment\1"
        output_directory = r"F:\Data\HX\Dataset\center1\Standardized\PostTreatment\Images_2"  # 可选

        # 处理文件
        results = split_sequences(
            input_directory,
            output_directory,
            recursive=True,
            keep_original=True
        )

        # 打印结果
        print_processing_results(results)

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
