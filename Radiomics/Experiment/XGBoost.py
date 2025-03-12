import numpy as np
import os
import optuna
import optunahub
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path
import logging

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
        self.feature_columns = None  # 存储训练集的特征列

    def load_data(self, excel_path: str, sheet_name: str = "Sheet1", is_training: bool = True) -> Tuple:
        """
        加载数据
        Args:
            excel_path: Excel文件路径
            sheet_name: 工作表名称
            is_training: 是否为训练数据集
        """
        data = pd.read_excel(excel_path, sheet_name=sheet_name)

        # 移除非特征列
        X = data.drop(["Target", "Patient"], axis=1, errors='ignore')
        X.columns = X.columns.astype(str)
        y = data["Target"]

        if is_training:
            # 如果是训练集，保存特征列名
            self.feature_columns = X.columns.tolist()
            logger.info(f"训练集特征数量: {len(self.feature_columns)}")
        else:
            # 如果是外部验证集，确保使用与训练集相同的特征
            if self.feature_columns is None:
                raise ValueError("必须先加载训练集才能加载外部验证集")

            # 检查是否缺失任何训练集的特征
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"外部验证集缺少以下特征: {missing_cols}")

            # 只选择训练集中存在的特征，并按相同顺序排列
            X = X[self.feature_columns]
            logger.info(f"外部验证集已对齐到训练集特征 (共{len(self.feature_columns)}个特征)")

        return X, y

    def load_external_data(self, excel_path: str, sheet_name: str = "Sheet1") -> Tuple:
        """
        加载外部验证数据集
        Args:
            excel_path: 外部验证数据集的Excel文件路径
            sheet_name: 工作表名称
        """
        # 使用load_data加载外部验证集，并指定is_training=False
        X_external, y_external = self.load_data(excel_path, sheet_name, is_training=False)

        # 使用与训练集相同的标准化器进行转换
        X_external_scaled = self.scaler.transform(X_external)

        return X_external_scaled, y_external

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """划分数据集"""
        if self.split_type == "binary":
            # 8:2划分
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=123
            )

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            return {
                'train': (X_train_scaled, y_train),
                'val': (X_val_scaled, y_val)
            }
        else:
            # 6:2:2划分
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=3407
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=3407
            )

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)

            return {
                'train': (X_train_scaled, y_train),
                'val': (X_val_scaled, y_val),
                'test': (X_test_scaled, y_test)
            }

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        if self.feature_columns is None:
            raise ValueError("未加载训练数据")
        return self.feature_columns


class XGBoostOptimizer:
    """XGBoost模型优化器"""

    def __init__(self, data_dict: Dict, study_name: str, storage_path: str):
        self.data_dict = data_dict
        self.study_name = study_name
        self.storage_path = storage_path
        self.best_params = None
        self.best_model = None
        self.feature_names = None

    def _objective(self, trial: optuna.Trial) -> float:
        """优化目标函数"""
        train_param = {
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'gamma': trial.suggest_float('gamma', 0.1, 2.0),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'alpha': trial.suggest_float('alpha', 0.1, 3.0),
            'lambda': trial.suggest_float('lambda', 0.1, 3.0),
            'objective': 'binary:logistic',  # 添加目标函数
            'eval_metric': ['auc', 'logloss']  # 添加评估指标
        }

        num_boost_round = trial.suggest_int('num_boost_round', 1000, 5000)
        early_stopping_rounds = 50

        X_train, y_train = self.data_dict['train']
        X_val, y_val = self.data_dict['val']
        # X_test, y_test = self.data_dict['test']

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        # dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            train_param,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False
        )

        # 使用DMatrix预测并返回验证集AUC
        y_pred = model.predict(dval)
        # y_test_pred = model.predict(dtest)

        return roc_auc_score(y_val, y_pred)

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

        # 保存最佳参数，包括最佳迭代次数
        self.best_params = {
            'train_params': {k: v for k, v in study.best_params.items() if k != 'num_boost_round'},
            'num_boost_round': study.best_params['num_boost_round']
        }

        # 训练最终模型
        self._train_final_model()

        return study

    def _train_final_model(self):
        """使用最佳参数训练最终模型"""
        X_train, y_train = self.data_dict['train']
        X_val, y_val = self.data_dict['val']

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # 添加目标函数和评估指标
        train_params = self.best_params['train_params']
        train_params.update({
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss']
        })

        # 使用早停法训练最终模型
        evals_result = {}
        self.best_model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=self.best_params['num_boost_round'],
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=True
        )

    def evaluate(self, external_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """
        评估模型性能
        Args:
            external_data: 可选的外部验证数据集，格式为 (X_external, y_external)
        """
        results = {}

        # 评估内部数据集
        for dataset_name, (X, y) in self.data_dict.items():
            y_pred_proba = self.best_model.predict(xgb.DMatrix(X))
            y_pred = (y_pred_proba > 0.5).astype(int)

            results[dataset_name] = {
                'auc': roc_auc_score(y, y_pred_proba),
                'classification_report': classification_report(y, y_pred),
                'predictions': {
                    'probabilities': y_pred_proba.tolist(),
                    'classes': y_pred.tolist()
                }
            }

        # 评估外部验证数据集
        if external_data is not None:
            X_external, y_external = external_data
            y_pred_proba_external = self.best_model.predict(xgb.DMatrix(X_external))
            y_pred_external = (y_pred_proba_external > 0.5).astype(int)

            results['external'] = {
                'auc': roc_auc_score(y_external, y_pred_proba_external),
                'classification_report': classification_report(y_external, y_pred_external),
                'predictions': {
                    'probabilities': y_pred_proba_external.tolist(),
                    'classes': y_pred_external.tolist()
                }
            }

        return results

    def save_model(self, model_path: str):
        """保存模型到文件"""
        self.best_model.save_model(model_path)

    def load_model(self, model_path: str):
        """从文件加载模型"""
        self.best_model = xgb.Booster()
        self.best_model.load_model(model_path)


def main():
    # 配置参数
    config = {
        'data_path': r"F:\Data\BreastClassification\Materials\ISPY2_CLINICAL.xlsx",
        'external_data_path': r"F:\Data\BreastClassification\Experiment\Radiomics\center1\Difference.xlsx",
        'split_type': 'binary',  # binary 或 triple
        'storage_path': r"..\Data\optuna-journal.log",
        'n_trials': 1000,
        'n_jobs': 8,
        'use_external_test': False
    }

    # 创建输出目录
    output_dir = Path(r"..\Data\20250116")
    output_dir.mkdir(exist_ok=True)

    # 初始化数据管理器和加载数据
    logger.info("加载和划分数据...")
    data_manager = DataManager(split_type=config['split_type'])
    X, y = data_manager.load_data(config['data_path'], sheet_name="Sheet2")
    data_dict = data_manager.split_data(X, y)

    # 加载外部验证数据集
    if config['use_external_test']:
        logger.info("加载外部验证数据集...")
        X_external, y_external = data_manager.load_external_data(config['external_data_path'])
        data_dict["test"] = (X_external, y_external)

    # 优化模型
    logger.info("开始模型优化...")
    optimizer = XGBoostOptimizer(
        data_dict=data_dict,
        study_name='xgboost_study',
        storage_path=config['storage_path']
    )
    study = optimizer.optimize(n_trials=config['n_trials'], n_jobs=config['n_jobs'])

    # 评估结果（包括外部验证）
    logger.info("评估模型性能...")
    if config['use_external_test']:
        results = optimizer.evaluate(external_data=(X_external, y_external))
    else:
        results = optimizer.evaluate()

    # 保存结果
    logger.info("保存结果...")
    for dataset_name, metrics in results.items():
        logger.info(f"\n{dataset_name} 集结果:")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"\n分类报告:\n{metrics['classification_report']}")

    # 保存最佳参数
    with open(output_dir / "best_params.json", 'w') as f:
        json.dump(optimizer.best_params, f, indent=4)

    # 保存模型
    model_path = output_dir / "best_model.json"
    optimizer.save_model(str(model_path))
    logger.info(f"最佳模型已保存到: {model_path}")

    # 保存预测结果
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"预测结果已保存到: {predictions_path}")

    logger.info("所有处理完成！")


if __name__ == "__main__":
    main()
