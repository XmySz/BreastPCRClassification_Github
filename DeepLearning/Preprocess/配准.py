import SimpleITK as sitk
import os
import numpy as np

def register_with_labels(fixed_image_path, moving_image_path, 
                        fixed_label_path, moving_label_path,
                        output_image_path, output_label_path,
                        use_label_as_mask=True):
    """
    使用标签信息进行3D图像配准
    
    Parameters:
    -----------
    fixed_image_path : str
        固定图像路径
    moving_image_path : str
        移动图像路径
    fixed_label_path : str
        固定图像标签路径
    moving_label_path : str
        移动图像标签路径
    output_image_path : str
        输出配准后的图像路径
    output_label_path : str
        输出配准后的标签路径
    use_label_as_mask : bool
        是否使用标签作为mask
    """
    # 读取图像和标签
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    fixed_label = sitk.ReadImage(fixed_label_path, sitk.sitkUInt8)
    moving_label = sitk.ReadImage(moving_label_path, sitk.sitkUInt8)
    
    # 初始化配准方法
    registration = sitk.ImageRegistrationMethod()
    
    # 设置度量方法
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # 如果使用标签作为mask
    if use_label_as_mask:
        # 创建mask（将所有非零标签区域设为1）
        fixed_mask = sitk.BinaryThreshold(fixed_label, 1, 255, 1, 0)
        registration.SetMetricMovingMask(fixed_mask)
    
    # 修改初始变换的设置
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    # 先进行初始对齐
    moving_image = sitk.Resample(moving_image, fixed_image, initial_transform, 
                                sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    moving_label = sitk.Resample(moving_label, fixed_label, initial_transform,
                                sitk.sitkNearestNeighbor, 0, moving_label.GetPixelID())
    
    # 重置变换为恒等变换
    registration.SetInitialTransform(sitk.Transform())
    
    # 修改优化器参数，使其更稳定
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10)
    
    # 修改多分辨率策略
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
    registration.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
    
    # 设置插值方法
    registration.SetInterpolator(sitk.sitkLinear)
    
    # 执行配准
    final_transform = registration.Execute(fixed_image, moving_image)
    
    # 应用变换到图像
    resampled_image = sitk.Resample(moving_image, 
                                   fixed_image, 
                                   final_transform, 
                                   sitk.sitkLinear,
                                   0.0,
                                   moving_image.GetPixelID())
    
    # 应用相同的变换到标签（使用最近邻插值以保持标签值）
    resampled_label = sitk.Resample(moving_label,
                                   fixed_label,
                                   final_transform,
                                   sitk.sitkNearestNeighbor,
                                   0,
                                   moving_label.GetPixelID())
    
    # 保存结果
    sitk.WriteImage(resampled_image, output_image_path)
    sitk.WriteImage(resampled_label, output_label_path)
    
    return final_transform

def register_folder_with_labels(fixed_dir, moving_dir, 
                              fixed_label_dir, moving_label_dir,
                              output_dir, output_label_dir):
    """
    对文件夹中的带标签的3D图像进行批量配准
    
    Parameters:
    -----------
    fixed_dir : str
        固定图像文件夹路径
    moving_dir : str
        移动图像文件夹路径
    fixed_label_dir : str
        固定图像标签文件夹路径
    moving_label_dir : str
        移动图像标签文件夹路径
    output_dir : str
        输出配准后图像的文件夹路径
    output_label_dir : str
        输出配准后标签的文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有图像和标签文件
    fixed_images = sorted([f for f in os.listdir(fixed_dir) if f.endswith(('.nii', '.nii.gz'))])
    moving_images = sorted([f for f in os.listdir(moving_dir) if f.endswith(('.nii', '.nii.gz'))])
    fixed_labels = sorted([f for f in os.listdir(fixed_label_dir) if f.endswith(('.nii', '.nii.gz'))])
    moving_labels = sorted([f for f in os.listdir(moving_label_dir) if f.endswith(('.nii', '.nii.gz'))])
    
    # 确保文件数量匹配
    assert len(fixed_images) == len(moving_images) == len(fixed_labels) == len(moving_labels), \
           "图像和标签数量不匹配"
    
    # 遍历所有图像对进行配准
    for i, (fixed_img, moving_img, fixed_lab, moving_lab) in enumerate(zip(
            fixed_images, moving_images, fixed_labels, moving_labels)):
        
        fixed_path = os.path.join(fixed_dir, fixed_img)
        moving_path = os.path.join(moving_dir, moving_img)
        fixed_label_path = os.path.join(fixed_label_dir, fixed_lab)
        moving_label_path = os.path.join(moving_label_dir, moving_lab)
        
        output_path = os.path.join(output_dir, f'registered_{moving_img}')
        output_label_path = os.path.join(output_label_dir, f'registered_{moving_lab}')
        
        print(f'正在配准第 {i+1}/{len(fixed_images)} 对图像: {moving_img}')
        try:
            transform = register_with_labels(fixed_path, moving_path,
                                          fixed_label_path, moving_label_path,
                                          output_path, output_label_path)
            print(f'配准完成: {moving_img}')
        except Exception as e:
            print(f'配准失败 {moving_img}: {str(e)}')

def calculate_dice_coefficient(label1_path, label2_path):
    """
    计算两个标签之间的Dice系数
    
    Parameters:
    -----------
    label1_path : str
        第一个标签文件路径
    label2_path : str
        第二个标签文件路径
        
    Returns:
    --------
    dict : 每个标签值的Dice系数
    """
    label1 = sitk.ReadImage(label1_path)
    label2 = sitk.ReadImage(label2_path)
    
    label1_array = sitk.GetArrayFromImage(label1)
    label2_array = sitk.GetArrayFromImage(label2)
    
    unique_labels = np.unique(np.concatenate([label1_array.flatten(), 
                                            label2_array.flatten()]))
    unique_labels = unique_labels[unique_labels != 0]  # 排除背景
    
    dice_scores = {}
    for label in unique_labels:
        intersection = np.sum((label1_array == label) & (label2_array == label))
        size1 = np.sum(label1_array == label)
        size2 = np.sum(label2_array == label)
        
        dice = 2.0 * intersection / (size1 + size2)
        dice_scores[int(label)] = dice
    
    return dice_scores

# 使用示例
if __name__ == "__main__":
    # 单文件配准示例
    fixed_image = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original\1062\T0\image.nii.gz"
    moving_image = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original\1062\T1\image.nii.gz"
    fixed_label = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original\1062\T0\label.nii.gz"
    moving_label = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original\1062\T1\label.nii.gz"
    output_image = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original\1062\T1\registered_image.nii.gz"
    output_label = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original\1062\T1\registered_label.nii.gz"
    
    transform = register_with_labels(fixed_image, moving_image,
                                   fixed_label, moving_label,
                                   output_image, output_label)
    
    # 计算配准后的Dice系数
    dice_scores = calculate_dice_coefficient(fixed_label, output_label)
    print("标签配准后的Dice系数:", dice_scores)
    
    # 文件夹配准示例
    # fixed_dir = "path/to/fixed_images"
    # moving_dir = "path/to/moving_images"
    # fixed_label_dir = "path/to/fixed_labels"
    # moving_label_dir = "path/to/moving_labels"
    # output_dir = "path/to/registered_images"
    # output_label_dir = "path/to/registered_labels"
    
    # register_folder_with_labels(fixed_dir, moving_dir,
    #                           fixed_label_dir, moving_label_dir,
    #                           output_dir, output_label_dir)