import glob
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_npz(file_path):
    """
    读取单个npz文件
    
    参数:
        file_path: npz文件路径
    返回:
        data_dict: 包含所有数组的字典
    """
    try:
        # 加载npz文件
        data = np.load(file_path)

        # 将数据转换为字典
        data_dict = {key: data[key] for key in data.files}

        print(f"成功读取文件: {file_path}")
        print(f"包含以下数组:")
        for key, array in data_dict.items():
            print(f"- {key}: 形状{array.shape}, 类型{array.dtype}")

        return data_dict

    except Exception as e:
        print(f"读取文件出错 {file_path}: {str(e)}")
        return None


def load_npy(file_path):
    """
    读取单个npy文件并显示数组信息
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
        print(f"images形状: {data['images'].shape}, 数据类型: {data['images'].dtype}")
        print(f"标签值: {data['label']}")
        return data

    except Exception as e:
        print(f"读取文件出错: {str(e)}")
        return None


def compare_label_voxels(resampled_dir, cropped_dir):
    """
    比较重采样和裁剪后的标签体素数量是否一致
    
    参数:
        resampled_dir: 重采样后的标签目录
        cropped_dir: 裁剪后的标签目录
    """
    print("开始比较标签体素数量...")

    # 统计不一致的数量
    mismatch_count = 0

    # 遍历重采样目录
    for root, dirs, files in os.walk(resampled_dir):
        for dir_name in dirs:
            # 构建标签文件路径
            resampled_label = os.path.join(root, dir_name, "label.nii.gz")
            cropped_label = os.path.join(cropped_dir, dir_name, "label.nii.gz")

            print(f"处理文件 {resampled_label}...")

            # 检查文件是否存在
            if not os.path.exists(resampled_label) or not os.path.exists(cropped_label):
                continue

            try:
                # 读取标签文件
                resampled_data = nib.load(resampled_label).get_fdata()
                cropped_data = nib.load(cropped_label).get_fdata()

                # 计算非零体素数量
                resampled_count = np.count_nonzero(resampled_data)
                cropped_count = np.count_nonzero(cropped_data)

                # 比较数量
                if resampled_count != cropped_count:
                    print(f"目录 {dir_name} 的标签体素数量不一致:")
                    print(f"  重采样: {resampled_count}")
                    print(f"  裁剪后: {cropped_count}")
                    print(f"  差异: {abs(resampled_count - cropped_count)}")
                    mismatch_count += 1

            except Exception as e:
                print(f"处理目录 {dir_name} 时出错: {str(e)}")

    print(f"\n比较完成. 共发现 {mismatch_count} 个不一致的标签")
    return mismatch_count


def process_nans(data_dir, output_dir=None):
    """处理数据集中的NaN值
    
    参数:
        data_dir (str): 输入数据目录路径
        output_dir (str): 输出数据目录路径,默认为None表示原地修改
    """
    print(f"\n开始处理目录 {data_dir} 中的数据文件...")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有npy文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    # 统计信息
    total_files = len(files)
    processed_files = 0

    # 使用tqdm创建进度条
    for file in tqdm(files, desc="处理文件"):
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path, allow_pickle=True).item()

        need_save = False

        # 处理图像数据中的NaN值
        if np.isnan(data['images']).any():
            print(f"\n发现文件 {file} 的图像数据中包含NaN值")
            # 使用0填充NaN值
            data['images'] = np.nan_to_num(data['images'], nan=0.0)
            need_save = True

        # 处理标签数据中的NaN值    
        if np.isnan(data['label']).any():
            print(f"\n发现文件 {file} 的标签数据中包含NaN值")
            # 标签中有NaN值的样本应该被移除
            print(f"警告: 文件 {file} 由于标签存在NaN值将被移除!")
            continue

        if need_save:
            processed_files += 1
            # 保存处理后的数据
            save_path = os.path.join(output_dir if output_dir else data_dir, file)
            np.save(save_path, data)
            print(f"已处理并保存到: {save_path}")

    print(f"\n处理完成!")
    print(f"总文件数: {total_files}")
    print(f"处理文件数: {processed_files}")


def check_label_gap(label_path, gap_threshold=30, distance_threshold=100):
    """
    检查标签文件中非零值层之间的间隔是否大于阈值,以及x或y轴上的最大距离是否超过阈值
    
    参数:
        label_path: str, 标签文件路径
        gap_threshold: int, 层间隔阈值,默认20
        distance_threshold: int, x/y轴距离阈值,默认150
        
    返回:
        bool: 如果存在大于阈值的间隔或距离返回True,否则返回False
        list: 可能存在问题的层数列表
    """
    try:
        # 读取标签文件
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()

        # 找到每一层中含有非零值的层索引
        nonzero_layers = []
        for i in range(label_data.shape[2]):
            if np.any(label_data[:, :, i] != 0):
                nonzero_layers.append(i)

        if len(nonzero_layers) < 2:
            return False, []

        # 检查相邻层之间的间隔
        problem_layers = []
        for i in range(len(nonzero_layers) - 1):
            gap = nonzero_layers[i + 1] - nonzero_layers[i]
            if gap > gap_threshold:
                problem_layers.extend([nonzero_layers[i], nonzero_layers[i + 1]])

        # 检查x和y轴上的最大距离
        for z in nonzero_layers:
            slice_data = label_data[:, :, z]
            nonzero_points = np.where(slice_data != 0)
            if len(nonzero_points[0]) > 0:
                x_distance = np.max(nonzero_points[0]) - np.min(nonzero_points[0])
                y_distance = np.max(nonzero_points[1]) - np.min(nonzero_points[1])
                max_distance = max(x_distance, y_distance)
                if max_distance > distance_threshold:
                    problem_layers.append(z)

        return [i + 1 for i in problem_layers]

    except Exception as e:
        print(f"处理文件出错 {label_path}: {str(e)}")
        return False, []


def replace_files(original_dir, replace_dir):
    """
    使用replace_dir中的文件替换original_dir中对应的文件
    
    参数:
        original_dir: 原始目录路径
        replace_dir: 包含替换文件的目录路径
    """
    print(f"开始替换文件...")
    print(f"原始目录: {original_dir}")
    print(f"替换文件目录: {replace_dir}")

    # 统计替换的文件数量
    replaced_count = 0

    # 递归遍历替换目录
    for root, dirs, files in os.walk(replace_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                # 获取相对路径
                rel_path = os.path.relpath(root, replace_dir)

                # 构建源文件和目标文件的完整路径
                src_file = os.path.join(root, file)
                dst_file = os.path.join(original_dir, rel_path, file)

                try:
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

                    # 复制并替换文件
                    shutil.copy2(src_file, dst_file)
                    replaced_count += 1
                    print(f"已替换: {dst_file}")

                except Exception as e:
                    print(f"替换文件 {dst_file} 时出错: {str(e)}")

    print(f"\n替换完成! 共替换了 {replaced_count} 个文件")


def get_roi_bounding_box(nii_path):
    """
    获取包含ROI的最小立方体信息

    参数:
        nii_path: str, nii.gz文件路径

    返回:
        tuple: (dimensions, points)
            dimensions: 最小立方体的三维尺寸 (x, y, z)
            points: 决定最大范围的坐标点 [(x_min,x_max), (y_min,y_max), (z_min,z_max)]
    """
    try:
        # 读取nii文件
        img = nib.load(nii_path)
        data = img.get_fdata()

        # 找到所有非零值的坐标
        coords = np.where(data > 0)

        if len(coords[0]) == 0:
            return None, None

        # 计算每个维度的最小最大值
        mins = np.min(coords, axis=1)
        maxs = np.max(coords, axis=1)

        # 计算立方体尺寸
        dimensions = maxs - mins + 1

        # 获取决定最大范围的坐标点
        points = []
        for i in range(3):
            min_idx = np.where(coords[i] == mins[i])[0][0]
            max_idx = np.where(coords[i] == maxs[i])[0][0]
            min_point = tuple(int(coords[j][min_idx]) + 1 for j in range(3))
            max_point = tuple(int(coords[j][max_idx]) + 1 for j in range(3))
            points.append((min_point, max_point))

        return tuple(int(d) for d in dimensions), points

    except Exception as e:
        print(f"处理文件出错 {nii_path}: {str(e)}")
        return None, None


def crop_by_bbox(image_path, mask_path, output_dir, margin=10):
    """
    根据mask的最小外接矩形体对图像和mask进行裁剪
    
    参数:
        image_path: str, 原始图像路径
        mask_path: str, mask图像路径
        output_dir: str, 输出目录
        margin: int, 裁剪时额外保留的边界范围,默认10个像素
    """
    try:
        # 读取图像和mask
        img = nib.load(image_path)
        mask = nib.load(mask_path)
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
        
        # 获取mask的边界框
        coords = np.where(mask_data > 0)
        if len(coords[0]) == 0:
            print(f"警告: {mask_path} 中没有非零值")
            return
            
        # 计算边界框范围(加上margin)
        mins = np.maximum(np.min(coords, axis=1) - margin, 0)
        maxs = np.minimum(np.max(coords, axis=1) + margin + 1, mask_data.shape)
        
        # 裁剪图像和mask
        cropped_img = img_data[int(mins[0]):int(maxs[0]), 
                             int(mins[1]):int(maxs[1]), 
                             int(mins[2]):int(maxs[2])]
        cropped_mask = mask_data[int(mins[0]):int(maxs[0]), 
                                int(mins[1]):int(maxs[1]), 
                                int(mins[2]):int(maxs[2])]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存裁剪后的图像
        cropped_img_nii = nib.Nifti1Image(cropped_img, img.affine)
        cropped_mask_nii = nib.Nifti1Image(cropped_mask, mask.affine)
        
        output_img_path = os.path.join(output_dir, "image.nii.gz")
        output_mask_path = os.path.join(output_dir, "label.nii.gz")
        
        nib.save(cropped_img_nii, output_img_path)
        nib.save(cropped_mask_nii, output_mask_path)
        
        print(f"裁剪完成:")
        print(f"原始形状: {img_data.shape}")
        print(f"裁剪后形状: {cropped_img.shape}")
        
    except Exception as e:
        print(f"裁剪出错: {str(e)}")


def z_score_normalize_directory(directory):
    """
    对指定目录下所有nii.gz文件执行z-score归一化

    参数:
        directory: str, 包含nii.gz文件的目录路径
    """
    # 获取目录下所有nii.gz文件
    nii_files = glob.glob(os.path.join(directory, "*.nii.gz"))
    
    print(f"\n开始对{len(nii_files)}个文件进行z-score归一化...")
    
    for file_path in tqdm(nii_files):
        try:
            # 读取nii文件
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # 计算均值和标准差
            mean = np.mean(data)
            std = np.std(data)
            
            # 执行z-score归一化
            if std != 0:
                normalized_data = (data - mean) / std
            else:
                print(f"警告: {file_path} 的标准差为0,跳过归一化")
                continue
                
            # 保存归一化后的文件
            normalized_img = nib.Nifti1Image(normalized_data, img.affine)
            nib.save(normalized_img, file_path)
            
        except Exception as e:
            print(f"处理文件出错 {file_path}: {str(e)}")
            continue
            
    print("\nz-score归一化完成!")


def calculate_intensity_stats(directory):
    """
    计算目录下所有nii.gz文件的强度值均值和标准差
    
    参数:
        directory: str, 包含nii.gz文件的目录路径
        
    返回:
        mean: float, 所有文件的平均强度值
        std: float, 所有文件的强度标准差
    """
    # 获取目录下所有nii.gz文件
    nii_files = glob.glob(os.path.join(directory, "**", "image.nii.gz"), recursive=True)[:100]
    
    if len(nii_files) == 0:
        print(f"警告: 目录 {directory} 中未找到nii.gz文件")
        return None, None
        
    print(f"\n开始统计{len(nii_files)}个文件的强度值...")
    
    # 收集所有像素值
    all_intensities = []
    
    for file_path in tqdm(nii_files):
        try:
            # 读取nii文件
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # 将数据展平并添加到列表中
            all_intensities.extend(data.flatten())
            
        except Exception as e:
            print(f"处理文件出错 {file_path}: {str(e)}")
            continue
    
    # 转换为numpy数组并计算统计值
    all_intensities = np.array(all_intensities)
    mean = np.mean(all_intensities)
    std = np.std(all_intensities)
    
    print(f"\n统计结果:")
    print(f"均值: {mean:.4f}")
    print(f"标准差: {std:.4f}")
    
    return mean, std


def rename_treatment_dirs(root_dir, excel_path):
    """
    重命名患者目录下的治疗时间点文件夹
    
    参数:
        root_dir: str, 根目录路径
        excel_path: str, 包含患者信息的Excel文件路径
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name='Sheet1')
    
    # 遍历每个患者记录
    for _, row in df.iterrows():
        patient_id = str(row['new_ID'])
        course = str(row['疗程'])
        
        # 构建患者目录路径
        patient_dir = os.path.join(root_dir, patient_id)
        
        if not os.path.exists(patient_dir):
            print(f"警告: 未找到患者目录 {patient_dir}")
            continue
            
        try:
            # 重命名pretreatment为T0
            pre_dir = os.path.join(patient_dir, 'pretreatment')
            if os.path.exists(pre_dir):
                os.rename(pre_dir, os.path.join(patient_dir, 'T0'))
                print(f"已将 {pre_dir} 重命名为 T0")
            
            # 根据疗程重命名treatment目录
            treat_dir = os.path.join(patient_dir, 'treatment')
            if os.path.exists(treat_dir):
                # 如果疗程是3或4则重命名为T2,否则为T1
                new_name = 'T2' if course in ['3', '4'] else 'T1'
                os.rename(treat_dir, os.path.join(patient_dir, new_name))
                print(f"已将 {treat_dir} 重命名为 {new_name}")
                
        except Exception as e:
            print(f"处理患者 {patient_id} 时出错: {str(e)}")
            continue
            
    print("重命名完成")


def split_timepoints(root_dir):
    """
    将多个时间点叠加的nii.gz文件分离为单独的时间点文件
    
    参数:
        root_dir: str, 根目录路径
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
                    
                    # 检查数据维度
                    if len(data.shape) > 3:
                        print(f"处理文件: {file_path}")
                        print(f"原始形状: {data.shape}")
                        
                        # 保留第一个时间点并去除多余维度
                        if len(data.shape) == 5:
                            data = data[..., 0, 0]
                        elif len(data.shape) == 4:
                            data = data[..., 0]
                            
                        # 创建新的nifti对象
                        new_img = nib.Nifti1Image(data, img.affine, img.header)
                        
                        # 保存文件
                        nib.save(new_img, file_path)
                        print(f"已保存处理后的文件,新形状: {data.shape}\n")
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}\n")
                    
    print("处理完成")


if __name__ == "__main__":
    crop_by_bbox(r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\NormalizedSpacing\1307\T3\image.nii.gz",
                 r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\NormalizedSpacing\1307\T3\label.nii.gz",
                 r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\NormalizedSpacing\1307\T3\cropped",
                 margin=5)
