import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import SimpleITK as sitk

class CropSizeAnalyzer:
    def __init__(self):
        self.image_shapes = []
        self.roi_sizes = []
        self.roi_centers = []
        
    def analyze_dataset(self, base_dir: str) -> None:
        """分析ISPY2数据集中的图像和ROI信息
        
        Args:
            base_dir: 数据集基础目录，包含所有患者子目录
        """
        base_path = Path(base_dir)
        
        # 遍历所有患者目录
        for patient_dir in base_path.glob("*"):
            if not patient_dir.is_dir():
                continue
                
            # 遍历每个患者的所有时间点
            for timepoint_dir in patient_dir.glob("*"):
                if not timepoint_dir.is_dir():
                    continue
                    
                # 构建图像和标签的完整路径
                image_path = timepoint_dir / "image.nii.gz"
                label_path = timepoint_dir / "label.nii.gz"
                
                if not image_path.exists() or not label_path.exists():
                    print(f"Warning: Missing files in {timepoint_dir}")
                    continue
                
                try:
                    # 读取原始图像尺寸
                    img = sitk.ReadImage(str(image_path))
                    self.image_shapes.append(img.GetSize())
                    
                    # 读取并分析标签
                    label = sitk.ReadImage(str(label_path))
                    label_array = sitk.GetArrayFromImage(label)
                    
                    if label_array.any():  # 如果标签不为空
                        bbox = self._get_bbox_3d(label_array)
                        self.roi_sizes.append(self._get_bbox_size(bbox))
                        self.roi_centers.append(self._get_bbox_center(bbox))
                        
                except Exception as e:
                    print(f"Error processing {timepoint_dir}: {str(e)}")
                    continue
        
        print(f"Successfully analyzed {len(self.image_shapes)} images")
        print(f"Found {len(self.roi_sizes)} valid ROIs")
    
    def _get_bbox_3d(self, binary_mask: np.ndarray) -> Tuple[slice, slice, slice]:
        """获取3D掩码的边界框"""
        r = np.any(binary_mask, axis=(1, 2))
        c = np.any(binary_mask, axis=(0, 2))
        z = np.any(binary_mask, axis=(0, 1))
        
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        
        return (slice(rmin, rmax + 1),
                slice(cmin, cmax + 1),
                slice(zmin, zmax + 1))
    
    def _get_bbox_size(self, bbox: Tuple[slice, slice, slice]) -> Tuple[int, int, int]:
        """计算边界框的尺寸"""
        return tuple(s.stop - s.start for s in bbox)
    
    def _get_bbox_center(self, bbox: Tuple[slice, slice, slice]) -> Tuple[int, int, int]:
        """计算边界框的中心点"""
        return tuple((s.stop + s.start) // 2 for s in bbox)
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if not self.image_shapes or not self.roi_sizes:
            raise ValueError("No data available. Please run analyze_dataset first.")
            
        stats = {
            'image_shapes': {
                'mean': np.mean(self.image_shapes, axis=0).tolist(),
                'std': np.std(self.image_shapes, axis=0).tolist(),
                'min': np.min(self.image_shapes, axis=0).tolist(),
                'max': np.max(self.image_shapes, axis=0).tolist(),
            },
            'roi_sizes': {
                'mean': np.mean(self.roi_sizes, axis=0).tolist(),
                'std': np.std(self.roi_sizes, axis=0).tolist(),
                'min': np.min(self.roi_sizes, axis=0).tolist(),
                'max': np.max(self.roi_sizes, axis=0).tolist(),
                '75th_percentile': np.percentile(self.roi_sizes, 75, axis=0).tolist(),
                '90th_percentile': np.percentile(self.roi_sizes, 90, axis=0).tolist(),
                '95th_percentile': np.percentile(self.roi_sizes, 95, axis=0).tolist(),
            }
        }
        return stats
    
    def suggest_crop_size(self, 
                         percentile: float = 95.0,
                         multiple_of: int = 16) -> Tuple[int, int, int]:
        """建议最佳裁剪尺寸
        
        Args:
            percentile: ROI尺寸的百分位数阈值
            multiple_of: 裁剪尺寸应该是此数的倍数(通常是2的幂次,适应下采样)
            
        Returns:
            建议的裁剪尺寸(d, h, w)
        """
        if not self.roi_sizes:
            raise ValueError("No ROI data available. Please run analyze_dataset first.")
            
        # 获取ROI尺寸的percentile分位数
        roi_size_threshold = np.percentile(self.roi_sizes, percentile, axis=0)
        
        # 向上取整到multiple_of的倍数
        crop_size = [int(np.ceil(s / multiple_of) * multiple_of) 
                    for s in roi_size_threshold]
        
        return tuple(crop_size)
    
    def visualize_distributions(self, save_path: str = None) -> None:
        """可视化图像和ROI尺寸的分布"""
        if not self.image_shapes or not self.roi_sizes:
            raise ValueError("No data available for visualization.")
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))
        
        # 图像尺寸分布
        image_data = [
            [shape[i] for shape in self.image_shapes]
            for i in range(3)
        ]
        axes[0].boxplot(image_data, labels=['Depth', 'Height', 'Width'])
        axes[0].set_title('Distribution of Image Shapes')
        axes[0].set_ylabel('Voxels')
        
        # 添加具体数值标注
        for i, data in enumerate(image_data, 1):
            mean_val = np.mean(data)
            axes[0].text(i, np.min(data), f'min: {int(np.min(data))}', 
                        horizontalalignment='center', verticalalignment='top')
            axes[0].text(i, np.max(data), f'max: {int(np.max(data))}', 
                        horizontalalignment='center', verticalalignment='bottom')
            axes[0].text(i, mean_val, f'mean: {int(mean_val)}', 
                        horizontalalignment='center', verticalalignment='bottom')
        
        # ROI尺寸分布
        roi_data = [
            [size[i] for size in self.roi_sizes]
            for i in range(3)
        ]
        axes[1].boxplot(roi_data, labels=['Depth', 'Height', 'Width'])
        axes[1].set_title('Distribution of ROI Sizes')
        axes[1].set_ylabel('Voxels')
        
        # 添加具体数值标注
        for i, data in enumerate(roi_data, 1):
            mean_val = np.mean(data)
            axes[1].text(i, np.min(data), f'min: {int(np.min(data))}', 
                        horizontalalignment='center', verticalalignment='top')
            axes[1].text(i, np.max(data), f'max: {int(np.max(data))}', 
                        horizontalalignment='center', verticalalignment='bottom')
            axes[1].text(i, mean_val, f'mean: {int(mean_val)}', 
                        horizontalalignment='center', verticalalignment='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution plot saved to {save_path}")
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = CropSizeAnalyzer()
    
    # 分析数据集
    base_dir = r"F:\Data\HX\Dataset\ISPY2_SELECT\Resampled"
    analyzer.analyze_dataset(base_dir)
    
    # 获取统计信息
    stats = analyzer.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # 获取建议的裁剪尺寸
    crop_size = analyzer.suggest_crop_size(percentile=95.0, multiple_of=16)
    print(f"\nSuggested crop size: {crop_size}")
    
    # 可视化分布
    analyzer.visualize_distributions("size_distributions.png")