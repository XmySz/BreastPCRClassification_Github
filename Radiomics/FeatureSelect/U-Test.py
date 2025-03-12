import os
import scipy.stats as stats
import pandas as pd
from pathlib import Path


def select_features(data: pd.DataFrame, alpha: float = 0.05):
    """特征选择主函数"""
    exclude_cols = ["Patient", "Target"]
    features = [col for col in data.columns if col not in exclude_cols]
    significant_features = []

    print(f"开始分析 {len(features)} 个特征...")

    for feature in features:
        group0 = data[data["Target"] == 0][feature]
        group1 = data[data["Target"] == 1][feature]

        try:
            _, pvalue = stats.mannwhitneyu(group0, group1, alternative='two-sided', nan_policy='omit')
            if pvalue < alpha:
                significant_features.append(feature)
                print(f"特征 {feature}: p值 = {pvalue:.4f} (显著)")
        except Exception as e:
            print(f"特征 {feature} 检验失败: {str(e)}")

    print(f"\n筛选出 {len(significant_features)} 个显著特征")
    return significant_features


def main(file_path: str):
    try:
        # 读取数据
        data = pd.read_excel(file_path, sheet_name="Sheet1")
        print(f"成功读取数据: {len(data)} 行")

        # 特征选择
        selected = select_features(data)

        # 保存结果
        output_path = Path(file_path).parent / f"{Path(file_path).stem}_U检验筛选.xlsx"
        result_df = data[["Patient", "Target"] + selected]
        result_df.to_excel(output_path, index=False)
        print(f"结果已保存至: {output_path}")

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    excel_path = r"F:\Data\HX\Experiment\Radiomics\Difference.xlsx"
    main(excel_path)