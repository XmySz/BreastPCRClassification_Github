import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from monai.transforms import (
    LoadImaged, SaveImaged, Compose, EnsureChannelFirstd, 
    SpatialCrop, Resized
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_bbox_center_and_size(label_array):
    """获取标签的边界框中心点和尺寸"""
    nonzero = np.array(np.nonzero(label_array))
    # 处理空标签的情况
    if nonzero.size == 0:
        # 返回图像中心点和默认尺寸
        return np.array([s // 2 for s in label_array.shape]), np.array([1, 1, 1])

    mins = np.min(nonzero, axis=1)
    maxs = np.max(nonzero, axis=1)
    center = ((maxs + mins) / 2).astype(int)
    size = (maxs - mins + 2)  # 加1确保完全包含ROI

    return center, size


class CropAroundLabeld:
    """在标签周围裁剪图像的变换类"""

    def __init__(self, keys, roi_size, allow_missing_keys=False):
        self.keys = keys
        self.roi_size = roi_size
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        # 获取标签中心点和ROI实际尺寸
        label_array = data['label'][0]  # 假设标签已经是channel first格式
        center, roi_actual_size = get_bbox_center_and_size(label_array)

        # 判断是否需要resize
        need_resize = any(a > r for a, r in zip(roi_actual_size, self.roi_size))

        if need_resize:
            # 如果ROI实际尺寸大于目标尺寸,先裁剪ROI区域再resize
            final_roi_size = [s + 4 for s in roi_actual_size]  
        else:
            # 否则使用目标尺寸进行裁剪
            final_roi_size = [max(r, a) for r, a in zip(self.roi_size, roi_actual_size)]

        # 计算裁剪的起始点，同时确保不会超出图像边界
        image_shape = data[self.keys[0]].shape[1:]  # 获取图像形状（不包括通道维度）
        start_coords = []
        for c, s, img_s in zip(center, final_roi_size, image_shape):
            start = max(0, min(c - s // 2, img_s - s))  # 确保结束点不会超出图像边界
            start_coords.append(start)

        # 使用SpatialCrop进行裁剪
        cropper = SpatialCrop(
            roi_start=start_coords,
            roi_end=[min(s + rs, img_s) for s, rs, img_s in zip(start_coords, final_roi_size, image_shape)]
        )

        # 对所有指定的键进行裁剪
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                data[key] = cropper(data[key])
                
                # 如果需要resize
                if need_resize:
                    resizer = Resized(
                        keys=[key],
                        spatial_size=self.roi_size,
                        mode="trilinear" if key == "image" else "nearest",
                        anti_aliasing=True,
                    )
                    data = resizer(data)
                    
        return data


def process_single_case(data_pair, transforms):
    """
    处理单个病例并进行错误处理
    """
    try:
        # 对数据对应用变换
        result = transforms(data_pair)
        return True, data_pair["image"], None
    except Exception as e:
        return False, data_pair["image"], str(e)


def process_batch(batch, transforms):
    """
    处理一批病例
    """
    results = []
    for data_pair in batch:
        success, filepath, error = process_single_case(data_pair, transforms)
        results.append((success, filepath, error))
    return results


def create_transforms(roi_size, save_root, data_root):
    """
    创建变换流水线
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropAroundLabeld(keys=["image", "label"], roi_size=roi_size),
        SaveImaged(
            keys=["image", "label"],
            output_dir=save_root,
            output_postfix="",
            separate_folder=False,
            data_root_dir=data_root,
            resample=False
        )
    ])


def process_dataset(
        data_root: str,
        save_root: str,
        roi_size=(128, 128, 32),
        batch_size=4,
        num_workers=4
):
    """
    使用并行处理对数据集进行ROI中心裁剪

    参数:
        data_root: 原始数据集根目录
        save_root: 处理后数据集保存目录
        roi_size: 裁剪目标大小
        batch_size: 每批处理的病例数
        num_workers: 工作进程数(默认: CPU核心数-1)
    """
    # 如果保存目录不存在则创建
    Path(save_root).mkdir(parents=True, exist_ok=True)

    # 设置工作进程数
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f"使用 {num_workers} 个进程进行并行处理")

    # 收集所有数据路径
    data_pairs = []
    for patient_id in os.listdir(data_root):
        patient_path = os.path.join(data_root, patient_id)
        if os.path.isdir(patient_path):
            for timepoint in os.listdir(patient_path):
                timepoint_path = os.path.join(patient_path, timepoint)
                if os.path.isdir(timepoint_path):
                    image_path = os.path.join(timepoint_path, "image.nii.gz")
                    label_path = os.path.join(timepoint_path, "label.nii.gz")
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        data_pairs.append({
                            "image": image_path,
                            "label": label_path
                        })

    # 创建变换
    transforms = create_transforms(roi_size, save_root, data_root)

    # 将数据分批
    total_cases = len(data_pairs)
    batches = [
        data_pairs[i:i + batch_size]
        for i in range(0, total_cases, batch_size)
    ]

    logger.info(f"正在处理 {total_cases} 个病例,共 {len(batches)} 批...")

    # 并行处理批次
    failed_cases = []
    processed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有批次
        future_to_batch = {
            executor.submit(process_batch, batch, transforms): batch
            for batch in batches
        }

        # 处理完成的结果
        for future in tqdm(as_completed(future_to_batch), total=len(batches)):
            batch = future_to_batch[future]
            try:
                results = future.result()
                for success, filepath, error in results:
                    if not success:
                        failed_cases.append((filepath, error))
                    processed_count += 1
            except Exception as e:
                logger.error(f"批处理失败: {str(e)}")
                for data_pair in batch:
                    failed_cases.append((data_pair["image"], str(e)))
                processed_count += len(batch)

    # 报告结果
    success_count = processed_count - len(failed_cases)
    logger.info(f"处理完成:")
    logger.info(f"成功处理: {success_count}/{total_cases} 个病例")

    if failed_cases:
        logger.warning(f"失败病例 ({len(failed_cases)}):")
        for filepath, error in failed_cases:
            logger.warning(f"- {filepath}: {error}")


def process_single_file(
        image_path: str,
        label_path: str,
        save_root: str,
        roi_size=(128, 128, 32)
):
    """
    处理单个文件的测试版本

    参数:
        image_path: 图像文件路径
        label_path: 标签文件路径  
        save_root: 保存目录
        roi_size: 裁剪目标大小
    """
    # 创建保存目录
    Path(save_root).mkdir(parents=True, exist_ok=True)

    # 准备数据对
    data_pair = {
        "image": image_path,
        "label": label_path
    }

    # 创建变换
    transforms = create_transforms(roi_size, save_root, os.path.dirname(image_path))

    # 处理数据
    try:
        result = transforms(data_pair)
        logger.info("处理成功!")
        return True
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 批量处理示例
    # data_root = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Original"
    # save_root = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Cropped"

    # process_dataset(
    #     data_root=data_root,
    #     save_root=save_root,
    #     roi_size=(128, 128, 96),
    #     batch_size=2,  # 根据系统内存调整
    #     num_workers=4  # 将使用CPU核心数-1
    # )

    # 单文件处理示例
    image_path = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Original\2976\T2\image.nii.gz"
    label_path = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Original\2976\T2\label.nii.gz"
    save_root = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Original\2976\T2\test"

    process_single_file(
        image_path=image_path,
        label_path=label_path,
        save_root=save_root,
        roi_size=(128, 128, 64)
    )
