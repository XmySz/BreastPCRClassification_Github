import os
import shutil
from pathlib import Path


def match_and_move_files():
    """
    匹配并移动文件
    
    从目录1读取文件,与目录2的文件进行匹配,将匹配的文件移动到目录3
    """
    # 定义目录路径
    dir1 = r"F:\Data\AdrenalNoduleClassification\Dataset\ExternalTest\Separate\SingleNodule\Images"
    dir2 = r"G:\LY\AdrenalClassification\Datasets\ExternalTest\Original\Labels"
    dir3 = r"G:\LY\AdrenalClassification\Datasets\ExternalTest\Original\S"
    # 创建目标目录
    Path(dir3).mkdir(parents=True, exist_ok=True)

    # 读取目录1的文件
    files1 = os.listdir(dir1)

    # 读取目录2的文件
    files2 = os.listdir(dir2)

    # 记录移动的文件数
    moved_count = 0

    print("开始匹配和移动文件...")

    # 遍历目录1的文件
    for file1 in files1:
        # 分割文件名获取第0部分和第2部分
        parts = file1.split('_')
        if len(parts) >= 3:
            match_pattern1 = f"{parts[0]}_{parts[2]}"

            # 在目录2中查找匹配的文件
            for file2 in files2:
                # 分割目录2中的文件名
                parts2 = file2.split('_')
                if len(parts2) >= 3:
                    match_pattern2 = f"{parts2[0]}_{parts2[2]}"

                    # 比较两个匹配模式
                    if match_pattern1 == match_pattern2:
                        # 构建源文件和目标文件路径
                        src_path = os.path.join(dir2, file2)
                        dst_path = os.path.join(dir3, file2)

                        try:
                            # 移动文件
                            shutil.move(src_path, dst_path)
                            moved_count += 1
                            print(f"已移动: {file2}")
                        except Exception as e:
                            print(f"移动文件 {file2} 时出错: {str(e)}")
    print(f"\n移动完成! 共移动了 {moved_count} 个文件")


if __name__ == "__main__":
    match_and_move_files()
