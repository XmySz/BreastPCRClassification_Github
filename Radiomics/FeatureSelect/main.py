import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector(ABC):
    """特征选择器基类"""

    def __init__(self, name: str):
        self.name = name
        self.selected_features: List[str] = []
        self.scores: Dict[str, float] = {}

    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """执行特征选择"""
        pass

    def get_feature_scores(self) -> Dict[str, float]:
        """获取特征得分"""
        return self.scores


class UTestSelector(FeatureSelector):
    """基于Mann-Whitney U检验的特征选择器"""

    def __init__(self, alpha: float = 0.05):
        super().__init__("U-Test")
        self.alpha = alpha

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        self.scores = {}
        for feature in X.columns:
            try:
                group0 = X[feature][y == 0]
                group1 = X[feature][y == 1]
                statistic, pvalue = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                self.scores[feature] = -np.log10(pvalue)  # 转换p值为分数
            except:
                self.scores[feature] = 0

        self.selected_features = [
            feature for feature, score in self.scores.items()
            if score > -np.log10(self.alpha)
        ]
        return self.selected_features


class LassoSelector(FeatureSelector):
    """基于LASSO的特征选择器"""

    def __init__(self, alpha: float = 1.0e-3):
        super().__init__("Lasso")
        self.alpha = alpha

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = Lasso(alpha=self.alpha, random_state=42)
        lasso.fit(X_scaled, y)

        self.scores = dict(zip(X.columns, np.abs(lasso.coef_)))
        self.selected_features = [
            feature for feature, coef in self.scores.items()
            if abs(coef) > 0
        ]
        return self.selected_features


class PearsonSelector(FeatureSelector):
    """基于Pearson相关系数的特征选择器"""

    def __init__(self, threshold: float = 0.8):
        super().__init__("Pearson")
        self.threshold = threshold

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        self.scores = {}
        for feature in X.columns:
            correlation = abs(stats.pearsonr(X[feature], y)[0])
            self.scores[feature] = correlation

        self.selected_features = [
            feature for feature, score in self.scores.items()
            if score > self.threshold
        ]
        return self.selected_features


class VarianceSelector(FeatureSelector):
    """基于方差的特征选择器"""

    def __init__(self, threshold: float = 0.01):
        super().__init__("Variance")
        self.threshold = threshold

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        # 计算每个特征的方差
        self.scores = dict(zip(X.columns, X.var()))

        # 使用sklearn的VarianceThreshold进行选择
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit_transform(X)

        # 获取选中的特征
        self.selected_features = X.columns[selector.get_support()].tolist()
        return self.selected_features


class CorrelationSelector(FeatureSelector):
    """基于相关性的特征选择器"""

    def __init__(self, threshold: float = 0.8):
        super().__init__("Correlation")
        self.threshold = threshold

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        # 计算相关性矩阵
        corr_matrix = X.corr().abs()

        # 获取上三角矩阵
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 找出高相关的特征
        high_corr_features = [
            column for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        # 为每个特征计算一个相关性得分（与其他特征的最大相关系数）
        self.scores = {}
        for col in X.columns:
            max_corr = corr_matrix[col].drop(col).max()
            self.scores[col] = 1 - max_corr  # 转换为分数（相关性越低分数越高）

        # 选择不在高相关列表中的特征
        self.selected_features = [
            feature for feature in X.columns
            if feature not in high_corr_features
        ]
        return self.selected_features


class RandomForestSelector(FeatureSelector):
    """基于随机森林的特征选择器"""

    def __init__(self, n_estimators: int = 100, threshold: float = 0.01):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.threshold = threshold

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42
        )
        rf.fit(X_scaled, y)

        # 获取特征重要性得分
        self.scores = dict(zip(X.columns, rf.feature_importances_))

        # 选择重要性超过阈值的特征
        self.selected_features = [
            feature for feature, importance in self.scores.items()
            if importance > self.threshold
        ]
        return self.selected_features


class FeatureSelectionPipeline:
    """特征选择流水线"""

    def __init__(self):
        self.selectors: List[FeatureSelector] = []
        self.final_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}

    def add_selector(self, selector: FeatureSelector) -> 'FeatureSelectionPipeline':
        """添加特征选择器"""
        self.selectors.append(selector)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int] = None) -> List[str]:
        """执行特征选择流程"""
        print(f"开始特征选择流程，原始特征数量: {len(X.columns)}")

        # 存储每个选择器的结果
        all_scores: Dict[str, Dict[str, float]] = {}

        # 运行所有选择器
        for selector in self.selectors:
            print(f"\n运行 {selector.name} 选择器...")
            selected = selector.select(X, y)
            scores = selector.get_feature_scores()
            all_scores[selector.name] = scores
            print(f"{selector.name} 选择的特征数量: {len(selected)}")

        # 综合所有选择器的结果
        self.feature_scores = {}
        for feature in X.columns:
            # 将每个选择器的分数归一化后求和
            normalized_scores = []
            for selector_name, scores in all_scores.items():
                if feature in scores:
                    score = scores[feature]
                    max_score = max(scores.values())
                    min_score = min(scores.values())
                    if max_score != min_score:
                        normalized_score = (score - min_score) / (max_score - min_score)
                        normalized_scores.append(normalized_score)

            if normalized_scores:
                self.feature_scores[feature] = np.mean(normalized_scores)

        # 根据综合得分排序
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 选择指定数量的特征
        if n_features is not None:
            self.final_features = [f[0] for f in sorted_features[:n_features]]
        else:
            self.final_features = [f[0] for f in sorted_features]

        print(f"\n最终选择的特征数量: {len(self.final_features)}")
        print("\nTop 10 特征及其得分:")
        for feature, score in sorted_features[:10]:
            print(f"{feature}: {score:.4f}")

        return self.final_features

    def save_selected_features(self,
                               original_data: pd.DataFrame,
                               output_path: str,
                               score_threshold: Optional[float] = None) -> None:
        """
        保存选定的特征到Excel文件

        Args:
            original_data: 原始数据DataFrame，包含Patient和Target列
            output_path: 输出Excel文件路径
            score_threshold: 特征分数阈值，如果设置，将选择分数高于此阈值的特征
        """
        # 确保Patient和Target列在原始数据中
        required_cols = ['Patient', 'Target']
        if not all(col in original_data.columns for col in required_cols):
            raise ValueError("原始数据必须包含 'Patient' 和 'Target' 列")

        # 如果设置了分数阈值，更新final_features
        if score_threshold is not None:
            self.final_features = [
                feature for feature, score in self.feature_scores.items()
                if score > score_threshold
            ]

        # 构建最终的列顺序
        final_columns = ['Patient'] + ['Target'] + self.final_features

        # 创建包含选定特征的新DataFrame
        selected_data = original_data[final_columns]

        try:
            # 保存到Excel
            selected_data.to_excel(output_path, index=False)
            print(f"\n选定特征已保存到: {output_path}")
            print(f"保存的特征数量: {len(self.final_features)}")
            print("列顺序: Patient, [选定特征], Target")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")


# 使用示例
def main():
    # 加载数据
    input_path = r"F:\Data\BreastClassification\Experiment\Radiomics\ISPY2_SELECT\PostTreatment.xlsx"
    output_path = r"F:\Data\BreastClassification\Experiment\Radiomics\ISPY2_SELECT\PostTreatmentSelected.xlsx"

    # 读取原始数据
    data = pd.read_excel(input_path)
    X = data.drop(['Target', 'Patient'], axis=1)
    y = data['Target']

    # 创建特征选择流水线
    pipeline = FeatureSelectionPipeline()

    # 添加特征选择器
    pipeline.add_selector(UTestSelector(alpha=0.05))
    pipeline.add_selector(LassoSelector(alpha=0.01))
    pipeline.add_selector(PearsonSelector(threshold=0.3))

    # 执行特征选择
    pipeline.fit(X, y, n_features=25)

    # 保存结果
    pipeline.save_selected_features(data, output_path)

    return pipeline.final_features


if __name__ == "__main__":
    main()
