import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple, Union


def preprocess_data(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    预处理数据，处理NaN值和类型转换

    Parameters:
    -----------
    y_pred : np.ndarray
        预测值数组
    y_true : np.ndarray
        真实标签数组

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        处理后的预测值和真实标签数组
    """
    # 转换为numpy数组
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    # 找出非NaN的索引
    valid_idx = ~(np.isnan(y_pred) | np.isnan(y_true))

    if not np.any(valid_idx):
        raise ValueError("所有数据都是NaN，无法计算指标")

    # 仅保留非NaN的值
    y_pred_clean = y_pred[valid_idx]
    y_true_clean = y_true[valid_idx]

    # 确保标签为0和1
    if not np.all(np.isin(y_true_clean, [0, 1])):
        raise ValueError("真实标签必须为0或1")

    return y_pred_clean, y_true_clean


def calculate_metrics(
        file_path: str,
        pred_column: str,
        label_column: str,
        sheet_name: Union[str, int] = 0,
        threshold: float = 0.5
) -> Tuple[float, float]:
    """
    从Excel或CSV文件中读取预测值和真实标签，计算AUC和ACC
    """
    try:
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 检查列是否存在
        if pred_column not in df.columns:
            raise ValueError(f"预测列 '{pred_column}' 不存在于数据中")
        if label_column not in df.columns:
            raise ValueError(f"标签列 '{label_column}' 不存在于数据中")

        # 预处理数据
        y_pred, y_true = preprocess_data(df[pred_column].values, df[label_column].values)

        # 计算指标
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred >= threshold)

        # 输出统计信息
        print("\n基础评估结果:")
        print(f"有效样本数: {len(y_true)}")
        print(f"正样本数: {sum(y_true == 1)}")
        print(f"负样本数: {sum(y_true == 0)}")
        print(f"NaN或无效样本数: {len(df) - len(y_true)}")
        print(f"AUC: {auc:.4f}")
        print(f"ACC: {acc:.4f}")

        return auc, acc

    except Exception as e:
        print(f"计算过程中出现错误: {str(e)}")
        raise


def calculate_metrics_with_ci(
        file_path: str,
        pred_column: str,
        label_column: str,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        sheet_name: Union[str, int] = 0,
        threshold: float = 0.5
) -> dict:
    """
    计算AUC和ACC，并通过bootstrap方法计算置信区间
    """
    try:
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 预处理数据
        y_pred, y_true = preprocess_data(df[pred_column].values, df[label_column].values)

        # 初始化bootstrap结果列表
        auc_boots = []
        acc_boots = []

        # 执行bootstrap
        n_samples = len(y_true)
        for _ in range(n_bootstrap):
            # 随机抽样（有放回）
            indices = np.random.randint(0, n_samples, n_samples)
            boot_pred = y_pred[indices]
            boot_true = y_true[indices]

            # 计算该bootstrap样本的指标
            auc_boots.append(roc_auc_score(boot_true, boot_pred))
            acc_boots.append(accuracy_score(boot_true, boot_pred >= threshold))

        # 计算置信区间
        alpha = (1 - confidence_level) / 2
        percentiles = [alpha * 100, (1 - alpha) * 100]

        auc_ci = np.percentile(auc_boots, percentiles)
        acc_ci = np.percentile(acc_boots, percentiles)

        # 计算原始指标
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred >= threshold)

        results = {
            'auc': auc,
            'acc': acc,
            'auc_ci': tuple(auc_ci),
            'acc_ci': tuple(acc_ci),
            'n_samples': n_samples,
            'n_positive': sum(y_true == 1),
            'n_negative': sum(y_true == 0),
            'n_invalid': len(df) - n_samples
        }

        # 打印详细结果
        print("\n详细评估结果:")
        print(f"有效样本总数: {results['n_samples']}")
        print(f"正样本数: {results['n_positive']}")
        print(f"负样本数: {results['n_negative']}")
        print(f"NaN或无效样本数: {results['n_invalid']}")
        print(f"AUC: {results['auc']:.4f} (95% CI: {results['auc_ci'][0]:.4f}-{results['auc_ci'][1]:.4f})")
        print(f"ACC: {results['acc']:.4f} (95% CI: {results['acc_ci'][0]:.4f}-{results['acc_ci'][1]:.4f})")

        return results

    except Exception as e:
        print(f"计算过程中出现错误: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    # 基础用法
    # try:
    #     auc, acc = calculate_metrics(
    #         file_path='your_data.xlsx',
    #         pred_column='prediction',
    #         label_column='label'
    #     )
    # except Exception as e:
    #     print(f"基础评估失败: {str(e)}")

    # 带置信区间的详细评估
    try:
        results = calculate_metrics_with_ci(
            file_path=r"F:\Data\BreastClassification\Materials\实验结果.xlsx",
            pred_column='model3_center1_predict',
            label_column='model3_center1_label',
            n_bootstrap=1000
        )
    except Exception as e:
        print(f"详细评估失败: {str(e)}")