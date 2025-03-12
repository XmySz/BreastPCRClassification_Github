import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import LoadImaged, SaveImaged, Compose, EnsureChannelFirstd, SpatialCrop
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_bbox_info(label_array):
    nonzero = np.array(np.nonzero(label_array))
    if nonzero.size == 0:
        return np.array([s // 2 for s in label_array.shape]), 32
    mins = np.min(nonzero, axis=1)
    maxs = np.max(nonzero, axis=1)
    center = ((maxs + mins) / 2).astype(int)
    return center, int(np.max(maxs - mins + 1))

class CropAroundROId:
    def __init__(self, keys, roi_size=(128, 128, 32), allow_missing_keys=False):
        self.keys = keys
        self.roi_size = roi_size
        self.allow_missing_keys = allow_missing_keys
        
    def __call__(self, data):
        label_array = data['label'][0]
        center, max_diameter = get_bbox_info(label_array)
        crop_size = [max_diameter * 2] * 3
        image_shape = data[self.keys[0]].shape[1:]
        start_coords = [max(0, min(c - s // 2, img_s - s)) 
                       for c, s, img_s in zip(center, crop_size, image_shape)]
        
        cropper = SpatialCrop(
            roi_start=start_coords,
            roi_end=[min(s + cs, img_s) 
                    for s, cs, img_s in zip(start_coords, crop_size, image_shape)]
        )
        
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                cropped = cropper(data[key])
                tensor = torch.from_numpy(cropped.array).unsqueeze(0)
                mode = "trilinear" if key == "image" else "nearest-exact"
                align_corners = False if key == "image" else None
                antialias = False if key == "image" else None
                
                resized = F.interpolate(
                    tensor, size=self.roi_size, mode=mode,
                    align_corners=align_corners, antialias=antialias
                ).squeeze(0).numpy()
                
                # 保留元数据
                data[key].array = resized
        return data

def process_single_file(image_path: str, label_path: str, save_root: str, roi_size=(64, 64, 64)):
    Path(save_root).mkdir(parents=True, exist_ok=True)
    transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropAroundROId(keys=["image", "label"], roi_size=roi_size),
        SaveImaged(
            keys=["image", "label"],
            output_dir=save_root,
            output_postfix="",
            separate_folder=False,
            data_root_dir=Path(image_path).parent,
            resample=False,
        )
    ])
    try:
        transforms({"image": image_path, "label": label_path})
        return True, None
    except Exception as e:
        return False, str(e)

def process_folder(data_root: str, save_root: str, roi_size=(64, 64, 64), num_workers=4):
    """处理整个文件夹的数据
    
    Args:
        data_root: 数据根目录,包含多个病例文件夹
        save_root: 保存根目录
        roi_size: 目标裁剪大小
        num_workers: 并行处理的线程数
    """
    data_root = Path(data_root)
    image_files = list(data_root.rglob("image.nii.gz"))
    total = len(image_files)
    failed_cases = []
    
    logger.info(f"找到 {total} 个病例")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for image_path in image_files:
            label_path = image_path.parent / "label.nii.gz"
            case_save_dir = Path(save_root) / image_path.parent.relative_to(data_root)
            futures.append(
                executor.submit(process_single_file, 
                              str(image_path), str(label_path), 
                              str(case_save_dir), roi_size)
            )
            
        for image_path, future in zip(image_files, tqdm(futures)):
            success, error = future.result()
            if not success:
                failed_cases.append((str(image_path), error))
                
    success_count = total - len(failed_cases)
    logger.info(f"处理完成: 成功 {success_count}/{total}")
    
    if failed_cases:
        logger.warning(f"失败病例 ({len(failed_cases)}):")
        for path, error in failed_cases:
            logger.warning(f"- {path}: {error}")

if __name__ == "__main__":
    # 单文件处理示例
    process_single_file(
        label_path=r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\NormalizedSpacing\1307\T3\cropped\label.nii.gz", 
        save_root=r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\NormalizedSpacing\1307\T3\croppedV2",
        roi_size=(64, 64, 64)
    )
    
    # 文件夹处理示例
    # process_folder(
    #     data_root=r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Original",
    #     save_root=r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\CroppedV2",
    #     roi_size=(64, 64, 64),
    #     num_workers=4
    # )