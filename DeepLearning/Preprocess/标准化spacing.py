from collections import OrderedDict
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    """
    重采样分割图像的专用函数
    通过将分割图转换为one-hot编码进行重采样,避免插值产生非法标签值

    参数:
        segmentation: np.ndarray, 输入的分割图
        new_shape: tuple, 目标尺寸
        order: int, 插值阶数
        cval: float, 边界填充值
    返回:
        np.ndarray: 重采样后的分割图
    """
    original_dtype = segmentation.dtype
    unique_labels = np.unique(segmentation)

    # 检查维度匹配
    assert len(segmentation.shape) == len(new_shape), "输入和目标形状的维度必须相同"

    # 零阶插值直接处理
    if order == 0:
        return resize(
            segmentation.astype(float),
            new_shape,
            order,
            mode="constant",
            cval=cval,
            clip=True,
            anti_aliasing=False
        ).astype(original_dtype)

    # 高阶插值需要对每个标签单独处理
    reshaped = np.zeros(new_shape, dtype=original_dtype)
    for label in unique_labels:
        mask = segmentation == label
        reshaped_multihot = resize(
            mask.astype(float),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False
        )
        reshaped[reshaped_multihot >= 0.5] = label

    return reshaped


def _process_2d_slice(data, slice_idx, axis, new_shape_2d, resize_fn, order, cval, kwargs):
    """
    处理单个2D切片的辅助函数

    参数:
        data: shape为(c, x, y, z)的4D数组
        slice_idx: 要处理的切片索引
        axis: 切片的轴向
        new_shape_2d: 目标2D形状
        resize_fn: 重采样函数
        order: 插值阶数
        cval: 填充值
        kwargs: 其他参数

    返回:
        重采样后的2D切片
    """
    if axis == 0:  # 沿x轴切片
        slice_data = data[:, slice_idx, :, :]
    elif axis == 1:  # 沿y轴切片
        slice_data = data[:, :, slice_idx, :]
    else:  # 沿z轴切片
        slice_data = data[:, :, :, slice_idx]

    # 对于单通道数据，去掉channel维度
    if slice_data.shape[0] == 1:
        slice_data = slice_data[0]
        reshaped = resize_fn(slice_data, new_shape_2d, order, cval=cval, **kwargs)
    else:
        # 多通道数据，每个通道单独处理
        reshaped = np.stack([
            resize_fn(slice_data[c], new_shape_2d, order, cval=cval, **kwargs)
            for c in range(slice_data.shape[0])
        ])

    return reshaped


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    """
    医学图像数据或分割图的重采样函数

    参数:
        data: np.ndarray, 输入数据, 格式为(c, x, y, z)
        new_shape: tuple, 目标形状
        is_seg: bool, 是否是分割数据
        axis: list, 各向异性轴的索引
        order: int, xy平面的插值阶数
        do_separate_z: bool, 是否单独处理z轴
        cval: float, 边界填充值
        order_z: int, z轴的插值阶数(仅当do_separate_z=True时有效)

    返回:
        np.ndarray: 重采样后的数据
    """
    # 基本检查和初始化
    assert len(data.shape) == 4, f"输入数据必须是4维的(c, x, y, z)，当前shape为: {data.shape}"

    resize_fn = resize_segmentation if is_seg else resize
    kwargs = OrderedDict() if is_seg else {'mode': 'edge', 'anti_aliasing': False}

    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data.shape[1:])  # 忽略channel维度
    new_shape = np.array(new_shape)

    # 如果形状相同则无需处理
    if np.all(shape == new_shape):
        print("形状相同,无需重采样")
        return data

    # 单独处理z轴
    if do_separate_z:
        print(f"单独处理z轴: z轴插值阶数={order_z}, 平面内插值阶数={order}")
        assert len(axis) == 1, "仅支持一个各向异性轴"
        axis = axis[0]
        print(f"Processing axis: {axis}")
        print(f"Input shape: {data.shape}")

        # 确定2D切片的新形状
        if axis == 0:
            new_shape_2d = new_shape[[1, 2]]
        elif axis == 1:
            new_shape_2d = new_shape[[0, 2]]
        else:  # axis == 2
            new_shape_2d = new_shape[[0, 1]]

        print(f"New 2D shape: {new_shape_2d}")

        reshaped_final_data = []
        for c in range(data.shape[0]):
            # 处理每个通道
            channel_data = data[c:c + 1]  # 保持4D格式，shape变为(1, x, y, z)
            print(f"Processing channel {c}, shape: {channel_data.shape}")

            # 处理每个切片
            reshaped_data = []
            for slice_id in range(shape[axis]):
                reshaped_slice = _process_2d_slice(
                    channel_data, slice_id, axis, new_shape_2d,
                    resize_fn, order, cval, kwargs
                )
                reshaped_data.append(reshaped_slice)

            reshaped_data = np.stack(reshaped_data, axis)

            # 如果z轴大小需要改变
            if shape[axis] != new_shape[axis]:
                print(f"Reshaped data shape before z interpolation: {reshaped_data.shape}")

                # 计算坐标映射
                if axis == 2:
                    coord_shape = (*new_shape_2d, new_shape[axis])
                    mgrid = np.mgrid[:coord_shape[0], :coord_shape[1], :coord_shape[2]]
                    scales = np.array([
                        reshaped_data.shape[0] / coord_shape[0],
                        reshaped_data.shape[1] / coord_shape[1],
                        reshaped_data.shape[2] / coord_shape[2]
                    ])[:, None, None, None]  # 添加维度以便广播
                elif axis == 1:
                    coord_shape = (new_shape[0], new_shape[axis], new_shape[2])
                    mgrid = np.mgrid[:coord_shape[0], :coord_shape[1], :coord_shape[2]]
                    scales = np.array([
                        reshaped_data.shape[0] / coord_shape[0],
                        reshaped_data.shape[1] / coord_shape[1],
                        reshaped_data.shape[2] / coord_shape[2]
                    ])[:, None, None, None]
                else:  # axis == 0
                    coord_shape = (new_shape[axis], new_shape[1], new_shape[2])
                    mgrid = np.mgrid[:coord_shape[0], :coord_shape[1], :coord_shape[2]]
                    scales = np.array([
                        reshaped_data.shape[0] / coord_shape[0],
                        reshaped_data.shape[1] / coord_shape[1],
                        reshaped_data.shape[2] / coord_shape[2]
                    ])[:, None, None, None]

                print(f"Coord shape: {coord_shape}")
                print(f"Scales shape: {scales.shape}")
                print(f"Mgrid shape: {mgrid.shape}")

                # 生成坐标映射
                coord_map = scales * (mgrid + 0.5) - 0.5

                # 进行z轴插值
                interpolated = _process_z_interpolation(
                    reshaped_data, coord_map, is_seg,
                    order_z, cval, dtype_data
                )
                reshaped_final_data.append(interpolated)
            else:
                reshaped_final_data.append(reshaped_data[None])

        reshaped_final_data = np.vstack(reshaped_final_data)
        print(f"Final resampled shape: {reshaped_final_data.shape}")

    # 整体处理
    else:
        print(f"整体重采样: 插值阶数={order}")
        reshaped = [
            resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None]
            for c in range(data.shape[0])
        ]
        reshaped_final_data = np.vstack(reshaped)

    return reshaped_final_data.astype(dtype_data)


def _process_z_interpolation(data, coord_map, is_seg, order_z, cval, dtype_data):
    """
    处理Z轴插值的辅助函数
    """
    print(f"Z interpolation - Data shape: {data.shape}")
    print(f"Z interpolation - Coord map shape: {coord_map.shape}")

    if not is_seg or order_z == 0:
        return map_coordinates(
            data,
            coord_map,
            order=order_z,
            cval=cval,
            mode='nearest'
        )[None]

    # 分割数据需要特殊处理每个标签
    unique_labels = np.unique(data)
    reshaped = np.zeros(coord_map.shape[1:], dtype=dtype_data)

    for label in unique_labels:
        mask = (data == label).astype(float)
        interpolated = map_coordinates(
            mask,
            coord_map,
            order=order_z,
            cval=cval,
            mode='nearest'
        )
        reshaped[np.round(interpolated) > 0.5] = label

    return reshaped[None]


def get_nifti_info(file_path: Path):
    """处理单个文件，返回spacing信息"""
    try:
        img = sitk.ReadImage(str(file_path))
        spacing = img.GetSpacing()
        size = img.GetSize()
        return file_path, spacing, size[:3]
    except Exception as e:
        print(f"处理文件出错 {file_path}: {e}")
        return None


def process_nifti_directory(directory: str, max_workers: int = 8):
    """处理指定目录下的所有.nii.gz文件"""
    # 获取所有.nii.gz文件路径
    root_dir = Path(directory)
    nifti_files = list(root_dir.rglob('*.nii.gz'))

    if not nifti_files:
        print("未找到.nii.gz文件")
        return

    print(f"找到 {len(nifti_files)} 个.nii.gz文件")

    # 并行处理所有文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(get_nifti_info, nifti_files)

    # 收集有效结果
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("没有成功处理的文件")
        return

    # 解析结果
    paths, spacings, shapes = zip(*valid_results)
    spacings_array = np.array(spacings)

    # 计算spacing中位数
    spacing_median = np.median(spacings_array, axis=0)

    # 计算并输出结果
    print(f"\n每个轴的spacing中位数: {spacing_median}")
    print("\n各文件信息:")

    for path, spacing, shape in zip(paths, spacings, shapes):
        # 计算新的shape
        scale_factors = np.array(spacing) / spacing_median
        new_shape = np.round(np.array(shape) * scale_factors).astype(int)

        # 输出信息
        print(f"\n文件: {path}")
        print(f"当前spacing: {spacing}")
        print(f"当前shape: {shape}")
        print(f"缩放后的新shape: {tuple(new_shape)}")


def process_single_file(
        src_file: Path,
        dst_file: Path,
        target_spacing: Tuple[float, float, float],
        max_threads_per_case: int = 8
) -> Optional[Tuple[Path, Tuple[float, float, float], Tuple[int, int, int]]]:
    """处理单个nii.gz文件的重采样"""
    try:
        # 设置numpy的随机种子以确保一致性
        np.random.seed(42)
        
        # 使用SimpleITK加载图像
        img = sitk.ReadImage(str(src_file))
        original_spacing = img.GetSpacing()[:3]
        data = sitk.GetArrayFromImage(img)
        
        # 转换数组维度顺序从(z,y,x)到(x,y,z)
        data = np.transpose(data, (2,1,0))
        
        # 使用更精确的浮点数类型
        scale_factors = np.array(original_spacing, dtype=np.float64) / np.array(target_spacing, dtype=np.float64)
        new_shape = np.round(np.array(data.shape[:3], dtype=np.float64) * scale_factors).astype(int)
        
        # 确保数据类型一致性
        data = data.astype(np.float32)  # 统一使用float32
        
        # 判断是否是分割图像
        is_seg = '/Images/' not in str(src_file)

        # 根据是否是分割图像设置重采样参数
        order = 0 if is_seg else 1  # 分割图像用最近邻插值
        do_separate_z = False  # 单独处理z轴以提高质量
        order_z = 0  # z轴始终使用最近邻插值

        # 准备数据格式 (c, x, y, z)
        if len(data.shape) == 3:
            data = data[None]  # 添加channel维度
        elif len(data.shape) == 4:
            # 检查第4维的大小
            if data.shape[3] == 1:
                # 如果第4维是1，去掉这个维度
                data = data[..., 0]
                data = data[None]
            else:
                # 如果第4维大于1，转置为(c, x, y, z)格式
                data = data.transpose(3, 0, 1, 2)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        print(f"Processing shape before resampling: {data.shape}")  # 添加调试信息

        # 执行重采样
        resampled_data = resample_data_or_seg(
            data=data,
            new_shape=new_shape,
            is_seg=is_seg,
            order=order,
            do_separate_z=do_separate_z,
            order_z=order_z,
            axis=[2]  # 假设z轴是第三个维度
        )

        print(f"Resampled shape: {resampled_data.shape}")  # 添加调试信息

        # 调整维度顺序回原格式
        if len(resampled_data.shape) == 4 and resampled_data.shape[0] > 1:
            resampled_data = resampled_data.transpose(1, 2, 3, 0)
        else:
            resampled_data = resampled_data[0]  # 去掉channel维度

        print(f"Final shape: {resampled_data.shape}")  # 添加调试信息

        # 转换数组维度顺序从(x,y,z)到(z,y,x)用于SimpleITK
        resampled_data = np.transpose(resampled_data, (2,1,0))
        
        # 创建SimpleITK图像
        output_image = sitk.GetImageFromArray(resampled_data)
        output_image.SetSpacing(target_spacing)
        output_image.SetDirection(img.GetDirection())
        output_image.SetOrigin(img.GetOrigin())

        # 保存重采样后的图像
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(output_image, str(dst_file))

        return src_file, original_spacing, new_shape

    except Exception as e:
        print(f"处理文件出错 {src_file}: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误堆栈:\n{traceback.format_exc()}")
        return None


def resample_dataset(
        source_dir: str,
        target_dir: str,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        max_workers: int = 4,
        max_threads_per_case: int = 8
):
    """
    重采样整个数据集，保持目录结构不变

    参数:
        source_dir: 源数据目录
        target_dir: 目标数据目录
        target_spacing: 可选的目标spacing元组 (x,y,z)。如果不指定,则自动计算数据集的中位数spacing
        max_workers: 同时处理的最大文件数
        max_threads_per_case: 每个文件处理时的最大线程数
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # 获取所有.nii.gz文件
    nifti_files = list(source_dir.rglob('*.nii.gz'))
    if not nifti_files:
        print("未找到.nii.gz文件")
        return

    print(f"找到 {len(nifti_files)} 个.nii.gz文件")

    # 如果没有指定target_spacing,则计算数据集的spacing中位数
    if target_spacing is None:
        print("未指定目标spacing,开始计算数据集的spacing中位数...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(get_nifti_info, nifti_files))

        valid_results = [r for r in results if r is not None]
        if not valid_results:
            print("没有成功处理的文件")
            return

        _, spacings, _ = zip(*valid_results)
        target_spacing = tuple(np.median(spacings, axis=0))
        print(f"计算得到的目标spacing (中位数): {target_spacing}")
    else:
        print(f"使用指定的目标spacing: {target_spacing}")

    # 准备重采样任务
    tasks = []
    for src_file in nifti_files:
        # 计算目标文件路径，保持相对路径结构
        rel_path = src_file.relative_to(source_dir)
        dst_file = target_dir / rel_path
        tasks.append((src_file, dst_file, target_spacing))

    # 执行重采样
    print("\n开始重采样...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, src, dst, spacing, max_threads_per_case)
            for src, dst, spacing in tasks
        ]

        # 收集结果并显示进度
        total = len(futures)
        for i, future in enumerate(futures, 1):
            try:
                result = future.result()
                if result:
                    src_file, original_spacing, new_shape = result
                    print(f"\n处理进度: [{i}/{total}]")
                    print(f"文件: {src_file}")
                    print(f"原始spacing: {original_spacing}")
                    print(f"重采样后shape: {new_shape}")
            except Exception as e:
                print(f"处理任务失败: {e}")

    print("\n重采样完成!")


if __name__ == "__main__":
    # 示例用法1: 自动计算目标spacing(原有功能)
    source_directory = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\Original"
    target_directory = r"F:\Data\BreastClassification\Dataset\ISPY2_SELECT\NormalizedSpacing"

    # resample_dataset(
    #     source_dir=source_directory,
    #     target_dir=target_directory,
    #     target_spacing=None,  # 不指定,自动计算
    #     max_workers=4,
    #     max_threads_per_case=4
    # )

    # # 示例用法2: 手动指定目标spacing
    resample_dataset(
        source_dir=source_directory,
        target_dir=target_directory,
        target_spacing=(0.7032, 0.7032, 2.0),  # 指定1mm各向同性spacing
        max_workers=4,
        max_threads_per_case=4
    )
