import os
import nibabel as nib
from pathlib import Path

def process_nifti_files(root_dir):
    """
    递归遍历目录处理nifti文件
    
    参数:
        root_dir: 根目录路径
    """
    print("开始处理文件...")
    
    # 遍历所有文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in ["image.nii.gz", "label.nii.gz"]:
                file_path = os.path.join(root, file)
                try:
                    # 加载nifti文件
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    
                    # 检查是否为5维数据
                    if len(data.shape) == 5:
                        print(f"处理文件: {file_path}")
                        print(f"原始形状: {data.shape}")
                        
                        # 保留第一个时间点
                        data = data[..., 0:1]
                        
                        # 创建新的nifti对象
                        new_img = nib.Nifti1Image(data, img.affine, img.header)
                        
                        # 保存文件
                        nib.save(new_img, file_path)
                        print(f"已保存处理后的文件,新形状: {data.shape}\n")
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}\n")

if __name__ == "__main__":
    root_dir = r"F:\Data\BreastClassification\Dataset\center1\Orginal"
    process_nifti_files(root_dir)
