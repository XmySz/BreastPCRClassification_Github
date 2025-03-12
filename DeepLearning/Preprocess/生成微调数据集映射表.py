import os
import random
from itertools import permutations

import pandas as pd


def is_continuous(sequence):
    """检查序列是否连续 (T0->T1->T2->T3)"""
    # 将Tx转换为数字
    nums = [int(s[1]) for s in sequence]
    # 检查是否为连续序列
    return sorted(nums) == nums


def get_combinations_with_length(items):
    """获取所有长度为4的排列组合并标记连续性"""
    perms = list(permutations(items, 4))
    continuous = []
    non_continuous = []

    for perm in perms:
        if is_continuous(perm):
            continuous.append(perm)
        else:
            non_continuous.append(perm)

    return continuous, non_continuous


def complete_to_four(items):
    """将少于4个元素的集合补充到4个元素"""
    all_possible = ['T0', 'T1', 'T2', 'T3']
    missing = [x for x in all_possible if x not in items]
    needed = 4 - len(items)
    return list(items) + random.sample(missing, needed)


def process_directory(root_dir):
    results = []

    # 遍历根目录下的所有子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            # 获取子目录中的T0-T3目录
            original_t_dirs = [d for d in os.listdir(subdir_path)
                               if os.path.isdir(os.path.join(subdir_path, d))
                               and d.startswith('T')]

            # 保存原始子目录列表
            original_dirs = '->'.join(sorted(original_t_dirs))

            # 补充到4个目录
            complete_t_dirs = complete_to_four(original_t_dirs)
            continuous_4, non_continuous_4 = get_combinations_with_length(complete_t_dirs)

            # 只有当连续和不连续的组合都存在时才处理
            if continuous_4 and non_continuous_4:
                # 选择组合
                selected = []
                # 添加一个连续组合作为正例
                selected.append({
                    'combination': random.choice(continuous_4),
                    'label': 1
                })
                # 随机选择3个不重复的不连续组合作为负例
                selected_non_continuous = random.sample(non_continuous_4, 3)
                for combo in selected_non_continuous:
                    selected.append({
                        'combination': combo,
                        'label': 0
                    })

                # 添加到结果中
                for combo in selected:
                    results.append({
                        'path': subdir_path,
                        'original_dirs': original_dirs,
                        'combination': '->'.join(combo['combination']),
                        'label': combo['label']
                    })

    df = pd.DataFrame(results)
    df.to_excel(r"F:\Data\HX\Dataset\ISPY2_SELECT\微调数据集映射表.xlsx", index=False)
    print(r"处理完成，结果已保存到 F:\Data\HX\Dataset\ISPY2_SELECT\微调数据集映射表.xlsx")
    return df


# 使用示例
if __name__ == "__main__":
    root_directory = r"F:\Data\HX\Dataset\ISPY2_SELECT\Cropped"
    df = process_directory(root_directory)
    print("\n结果预览：")
    print(df)
