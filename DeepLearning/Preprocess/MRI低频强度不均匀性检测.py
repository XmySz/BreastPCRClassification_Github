import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
from scipy.stats import variation

def detect_intensity_inhomogeneity(image_path, slice_idx=None):
    """
    检测MRI图像中的低频强度不均匀性
    
    参数:
    image_path: str, NIfTI格式MRI图像路径
    slice_idx: int, 可选, 要分析的切片索引
    
    返回:
    dict: 包含多个评估指标的字典
    """
    # 加载图像
    img = nib.load(image_path)
    data = img.get_fdata()
    
    # 如果未指定切片,使用中间切片
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    
    # 获取选定切片
    slice_data = data[:, :, slice_idx]
    
    # 1. 计算变异系数 (CV)
    def calculate_cv(img, window_size=50):
        """计算局部区域的变异系数"""
        windows = []
        h, w = img.shape
        for i in range(0, h-window_size, window_size//2):
            for j in range(0, w-window_size, window_size//2):
                window = img[i:i+window_size, j:j+window_size]
                if window.size > 0:
                    cv = variation(window.flatten())
                    windows.append(cv)
        return np.mean(windows), np.std(windows)
    
    cv_mean, cv_std = calculate_cv(slice_data)
    
    # 2. 计算平滑度指标
    def calculate_smoothness(img):
        """计算图像平滑度"""
        gradient_x = ndimage.sobel(img, axis=0)
        gradient_y = ndimage.sobel(img, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.mean(gradient_magnitude)
    
    smoothness = calculate_smoothness(slice_data)
    
    # 3. 计算中心-边缘比
    def calculate_center_edge_ratio(img):
        """计算中心区域和边缘区域的强度比"""
        h, w = img.shape
        center_region = img[h//4:3*h//4, w//4:3*w//4]
        edge_region = img.copy()
        edge_region[h//4:3*h//4, w//4:3*w//4] = 0
        
        center_mean = np.mean(center_region[center_region != 0])
        edge_mean = np.mean(edge_region[edge_region != 0])
        
        return center_mean / edge_mean if edge_mean != 0 else float('inf')
    
    center_edge_ratio = calculate_center_edge_ratio(slice_data)
    
    # 4. 低频分量分析
    def analyze_low_frequency(img):
        """分析图像的低频分量"""
        # 使用高斯滤波获取低频分量
        low_freq = ndimage.gaussian_filter(img, sigma=5)
        # 计算低频分量与原图的比值
        ratio = np.std(low_freq) / np.std(img)
        return ratio
    
    low_freq_ratio = analyze_low_frequency(slice_data)
    
    # 汇总结果
    results = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'smoothness': smoothness,
        'center_edge_ratio': center_edge_ratio,
        'low_freq_ratio': low_freq_ratio
    }
    
    # 判断是否存在明显的不均匀性
    def assess_inhomogeneity(results):
        """评估是否存在明显的不均匀性"""
        criteria = {
            'cv_high': results['cv_mean'] > 0.2,  # 变异系数过高
            'smooth_high': results['smoothness'] < 0.1,  # 过度平滑
            'ratio_high': abs(results['center_edge_ratio'] - 1) > 0.2,  # 中心-边缘比差异大
            'low_freq_high': results['low_freq_ratio'] > 0.5  # 低频分量占比大
        }
        
        return criteria, sum(criteria.values()) >= 2
    
    criteria, has_inhomogeneity = assess_inhomogeneity(results)
    results['has_inhomogeneity'] = has_inhomogeneity
    results['criteria'] = criteria
    
    # 可视化结果
    def visualize_results(img, results):
        """可视化分析结果"""
        plt.figure(figsize=(12, 4))
        
        # 原始图像
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # 低频分量
        plt.subplot(132)
        low_freq = ndimage.gaussian_filter(img, sigma=5)
        plt.imshow(low_freq, cmap='gray')
        plt.title('Low Frequency Component')
        plt.axis('off')
        
        # 中心-边缘比可视化
        plt.subplot(133)
        h, w = img.shape
        mask = np.zeros_like(img)
        mask[h//4:3*h//4, w//4:3*w//4] = 1
        plt.imshow(img * mask, cmap='gray')
        plt.title(f'Center/Edge Ratio: {results["center_edge_ratio"]:.2f}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    visualize_results(slice_data, results)
    
    return results

# 使用示例
def main():
    """主函数"""
    image_path = r"F:\Data\HX\Dataset\ISPY2_SELECT\Resampled\1062\T2\image.nii.gz"  # 替换为实际的图像路径
    results = detect_intensity_inhomogeneity(image_path)
    
    print("\nIntensity Inhomogeneity Analysis Results:")
    print("-" * 40)
    print(f"变异系数 (CV) 均值: {results['cv_mean']:.3f}")
    print(f"变异系数 (CV) 标准差: {results['cv_std']:.3f}")
    print(f"平滑度指标: {results['smoothness']:.3f}")
    print(f"中心-边缘比: {results['center_edge_ratio']:.3f}")
    print(f"低频分量比例: {results['low_freq_ratio']:.3f}")
    print(f"是否存在明显的不均匀性: {'是' if results['has_inhomogeneity'] else '否'}")
    
    print("\n具体判断标准:")
    for criterion, value in results['criteria'].items():
        print(f"{criterion}: {'不通过' if value else '通过'}")

if __name__ == "__main__":
    main()