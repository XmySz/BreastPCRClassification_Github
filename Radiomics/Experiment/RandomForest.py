import numpy as np
import os
import optuna
import optunahub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path
import logging
import joblib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataManager:
    """数据管理类"""

    def __init__(self, split_type: str = "binary"):
        """
        初始化数据管理器
        Args:
            split_type: 'binary' (8:2) 或 'triple' (6:2:2)
        """
        self.split_type = split_type
        self.scaler = StandardScaler()
        self.feature_columns = None  # 存储训练数据的特征列

    def load_data(self, excel_path: str) -> Tuple:
        """加载数据"""
        data = pd.read_excel(excel_path, sheet_name="Sheet1")
        X = data.drop(["Target", "Patient"], axis=1)
        X.columns = X.columns.astype(str)
        self.feature_columns = X.columns  # 保存特征列名
        y = data["Target"]
        return X, y

    def load_external_data(self, excel_path: str) -> Tuple:
        """
        加载外部验证数据，并确保特征列与训练数据匹配
        Args:
            excel_path: 外部验证数据的路径
        Returns:
            处理后的特征矩阵和标签
        """
        if self.feature_columns is None:
            raise ValueError("必须先加载训练数据才能加载外部验证数据")

        # 加载外部数据
        data = pd.read_excel(excel_path, sheet_name="Sheet1")

        # 检查必要的列
        if "Target" not in data.columns:
            raise ValueError("外部验证数据中缺少'Target'列")

        # 提取特征和标签
        y_external = data["Target"]

        # 创建一个与训练集特征相同的DataFrame，初始值为0
        X_external = pd.DataFrame(0, index=data.index, columns=self.feature_columns)

        # 复制匹配的列
        common_cols = set(data.columns) & set(self.feature_columns)
        for col in common_cols:
            if col not in ["Target", "Patient"]:
                X_external[col] = data[col]

        # 记录缺失的特征
        missing_features = set(self.feature_columns) - common_cols
        if missing_features:
            logger.warning(f"外部验证集缺少以下特征（已用0填充）: {missing_features}")

        # 记录额外的特征
        extra_features = set(data.columns) - set(["Target", "Patient"]) - set(self.feature_columns)
        if extra_features:
            logger.warning(f"外部验证集含有额外的特征（已忽略）: {extra_features}")

        return X_external, y_external

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """划分数据集"""
        if self.split_type == "binary":
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=3407
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            return {
                'train': (X_train_scaled, y_train),
                'val': (X_val_scaled, y_val)
            }
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=3407
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=3407
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)

            return {
                'train': (X_train_scaled, y_train),
                'val': (X_val_scaled, y_val),
                'test': (X_test_scaled, y_test)
            }

    def transform_external_data(self, X_external: pd.DataFrame) -> np.ndarray:
        """
        使用训练集的缩放器转换外部验证数据
        """
        if self.scaler is None:
            raise ValueError("必须先训练数据才能转换外部验证数据")

        return self.scaler.transform(X_external)


class RandomForestOptimizer:
    """随机森林模型优化器"""

    def __init__(self, data_dict: Dict, study_name: str, storage_path: str, external_data):
        self.data_dict = data_dict
        self.study_name = study_name
        self.storage_path = storage_path
        self.best_params = None
        self.best_model = None
        self.feature_importances_ = None
        self.external_data = external_data

    def _objective(self, trial: optuna.Trial) -> float:
        """优化目标函数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': 3407
        }

        X_train, y_train = self.data_dict['train']

        # 训练模型
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 预测验证集
        # X_val, y_val = self.data_dict['val']
        # y_pred_proba = model.predict_proba(X_val)[:, 1]
        # return roc_auc_score(y_val, y_pred_proba)

        # 预测验证集
        y_test_proba = model.predict_proba(self.external_data[0])[:, 1]
        return roc_auc_score(self.external_data[1], y_test_proba)

    def optimize(self, n_trials: int = 1500, n_jobs: int = 8):
        """运行优化"""
        # 清理旧的存储文件
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        if os.path.exists(self.storage_path + ".lock"):
            os.remove(self.storage_path + ".lock")

        # 创建存储和研究
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(self.storage_path)
        )

        auto_sampler_module = optunahub.load_module("samplers/auto_sampler")
        study = optuna.create_study(
            sampler=auto_sampler_module.AutoSampler(),
            direction='maximize',
            study_name=self.study_name,
            storage=storage,
        )

        # 运行优化
        study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs)

        # 保存最佳参数
        self.best_params = study.best_params

        # 训练最终模型
        self._train_final_model()

        return study

    def _train_final_model(self):
        """使用最佳参数训练最终模型"""
        X_train, y_train = self.data_dict['train']

        # 训练最终模型
        self.best_model = RandomForestClassifier(**self.best_params)
        self.best_model.fit(X_train, y_train)

        # 保存特征重要性
        self.feature_importances_ = pd.Series(
            self.best_model.feature_importances_,
            index=range(X_train.shape[1])  # 使用特征索引作为名称
        ).sort_values(ascending=False)

    def evaluate(self, external_data: Optional[Tuple] = None) -> Dict:
        """
        评估模型性能，包括外部验证数据
        Args:
            external_data: 可选的外部验证数据元组 (X_external_scaled, y_external)
        """
        results = {}

        # 评估训练和验证/测试集
        for dataset_name, (X, y) in self.data_dict.items():
            y_pred_proba = self.best_model.predict_proba(X)[:, 1]
            y_pred = self.best_model.predict(X)

            results[dataset_name] = {
                'auc': roc_auc_score(y, y_pred_proba),
                'classification_report': classification_report(y, y_pred),
                'predictions': {
                    'probabilities': y_pred_proba,
                    'classes': y_pred
                }
            }

        # 评估外部验证集
        if external_data is not None:
            X_external_scaled, y_external = external_data
            y_pred_proba = self.best_model.predict_proba(X_external_scaled)[:, 1]
            y_pred = self.best_model.predict(X_external_scaled)

            results['external'] = {
                'auc': roc_auc_score(y_external, y_pred_proba),
                'classification_report': classification_report(y_external, y_pred),
                'predictions': {
                    'probabilities': y_pred_proba,
                    'classes': y_pred
                }
            }

        return results

    def save_model(self, path: Path):
        """保存模型和特征重要性"""
        if not path.exists():
            path.mkdir(parents=True)

        # 保存模型
        joblib.dump(self.best_model, path / 'best_model.joblib')

        # 保存特征重要性
        self.feature_importances_.to_csv(path / 'feature_importances.csv')


def main():
    # 配置参数
    config = {
        'data_path': r"F:\Data\HX\Experiment\Radiomics\ISPY2_SELECT\DifferenceSelected_1.xlsx",
        'external_data_path': r"F:\Data\HX\Experiment\Radiomics\center1\Difference.xlsx",  # 外部验证数据路径
        'split_type': 'binary',  # binary 或 triple
        'storage_path': r"..\Data\optuna-journal.log",
        'n_trials': 1000,
        'n_jobs': 8
    }

    # 创建输出目录
    output_dir = Path(r"..\Data\20241113_RF")
    output_dir.mkdir(exist_ok=True)

    # 初始化数据管理器和加载数据
    logger.info("加载和划分数据...")
    data_manager = DataManager(split_type=config['split_type'])

    # 加载训练数据
    X, y = data_manager.load_data(config['data_path'])
    data_dict = data_manager.split_data(X, y)

    # 加载外部验证数据（如果存在）
    external_data = None
    if 'external_data_path' in config and os.path.exists(config['external_data_path']):
        logger.info("加载外部验证数据...")
        try:
            X_external, y_external = data_manager.load_external_data(config['external_data_path'])
            X_external_scaled = data_manager.transform_external_data(X_external)
            external_data = (X_external_scaled, y_external)
        except Exception as e:
            logger.error(f"加载外部验证数据时出错: {str(e)}")

    # 优化模型
    logger.info("开始模型优化...")
    optimizer = RandomForestOptimizer(
        data_dict=data_dict,
        study_name='random_forest_study',
        storage_path=config['storage_path'],
        external_data=external_data,
    )
    study = optimizer.optimize(n_trials=config['n_trials'], n_jobs=config['n_jobs'])

    # 评估结果
    logger.info("评估模型性能...")
    results = optimizer.evaluate(external_data)

    # 保存结果
    logger.info("保存结果...")
    for dataset_name, metrics in results.items():
        logger.info(f"\n{dataset_name} 集结果:")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"\n分类报告:\n{metrics['classification_report']}")

    # 保存最佳参数
    with open(output_dir / "best_params.json", 'w') as f:
        json.dump(optimizer.best_params, f, indent=4)

    # 保存模型和特征重要性
    optimizer.save_model(output_dir)

    logger.info(f"最佳参数和模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()