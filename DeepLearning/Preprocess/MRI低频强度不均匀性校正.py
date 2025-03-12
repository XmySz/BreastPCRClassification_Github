import SimpleITK as sitk
import os
from pathlib import Path


def n4_bias_correction(input_image_path, output_image_path, mask_image_path=None):
    """
    对单个MRI图像进行N4偏置场校正
    
    参数:
        input_image_path: 输入图像路径
        output_image_path: 输出图像保存路径 
        mask_image_path: 可选的掩码图像路径
    """
    # 读取输入图像
    image = sitk.ReadImage(input_image_path)

    # 将图像转换为float32类型
    image = sitk.Cast(image, sitk.sitkFloat32)

    # 如果提供了mask图像则读取
    mask_image = None
    if mask_image_path is not None:
        mask_image = sitk.ReadImage(mask_image_path)
        mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)

    # 创建N4偏置场校正器
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # 设置参数
    corrector.SetMaximumNumberOfIterations([50] * 4)  # 4个分辨率层次,每层50次迭代
    corrector.SetConvergenceThreshold(0.001)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrector.SetWienerFilterNoise(0.01)
    corrector.SetNumberOfHistogramBins(200)
    corrector.SetNumberOfControlPoints([4] * 3)  # 3D图像的三个维度各4个控制点

    try:
        # 执行校正
        output_image = corrector.Execute(image, mask_image)

        # 保存结果
        sitk.WriteImage(output_image, output_image_path)
        print(f"成功处理图像: {input_image_path}")

    except Exception as e:
        print(f"处理图像 {input_image_path} 时出错: {str(e)}")


def process_directory(input_dir, output_dir, file_extension='.nii.gz'):
    """
    处理目录下的所有MRI图像
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        file_extension: 图像文件扩展名
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有符合扩展名的文件
    input_files = list(Path(input_dir).glob(f'*{file_extension}'))

    if not input_files:
        print(f"在 {input_dir} 中未找到 {file_extension} 文件")
        return

    # 处理每个文件
    for input_file in input_files:
        output_file = Path(output_dir) / f"{input_file.stem}_corrected{file_extension}"
        n4_bias_correction(str(input_file), str(output_file))


def main():
    input = r"F:\Data\HX\Dataset\ISPY2_SELECT\Resampled\1586\T2\image.nii.gz"
    output = r"F:\Data\HX\Dataset\ISPY2_SELECT\Resampled\1586\T2\image_corrected.nii.gz"
    mask = r"F:\Data\HX\Dataset\ISPY2_SELECT\Resampled\1586\T2\label.nii.gz"

    n4_bias_correction(input, output, mask)


if __name__ == '__main__':
    main()
