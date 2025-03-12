import concurrent.futures
import glob
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    def __init__(self):
        self.settings = {
            'binWidth': 25,
            'normalize': True,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': [1, 1, 1],
            'correctMask': True,
        }
        self.extractor = self._initialize_extractor()

    def _initialize_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        """初始化特征提取器"""
        extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)
        extractor.enableAllFeatures()
        extractor.enableImageTypeByName('Original')
        extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0]})
        extractor.enableImageTypeByName('Wavelet')
        return extractor

    def extract_features(self, case_data: Tuple[str, str, str]) -> Tuple[str, List[str], List[float]]:
        """提取单个病例的影像组学特征"""
        image_path, mask_path, patient = case_data
        logger.info(f"正在处理病例: {patient}")

        try:
            image = sitk.ReadImage(image_path, sitk.sitkUInt8)
            label = sitk.ReadImage(mask_path, sitk.sitkUInt8)
            result = self.extractor.execute(image, label, label=1)
            filtered_result = {k: v for i, (k, v) in enumerate(result.items()) if i >= 37}
            feature_names = list(filtered_result.keys())
            feature_values = list(filtered_result.values())
            return patient, feature_names, feature_values

        except Exception as e:
            logger.error(f"处理病例 {patient} 时发生错误: {str(e)}")
            return patient, [], []


def get_file_paths(base_dir: str, pattern: str) -> List[str]:
    """获取指定目录下符合模式的文件路径"""
    return sorted(glob.glob(os.path.join(base_dir, pattern)))


def save_features(df: pd.DataFrame, output_path: str):
    """保存特征到Excel文件"""
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_excel(output_path, index=False)
        logger.info(f"特征已成功保存到文件: {output_path}")
    except Exception as e:
        logger.error(f"保存特征时发生错误: {str(e)}")


def main():
    # 配置路径
    config = {
        'image_dir': r"F:\Data\BreastClassification\Dataset\center2\Normalized\PreTreatment\Images",
        'mask_dir':  r"F:\Data\BreastClassification\Dataset\center2\Normalized\PreTreatment\Labels",
        'output_path': r"F:\Data\BreastClassification\Dataset\center2\Normalized\PreTreatment.xlsx",
        'num_workers': 4,
        'sample_limit': None  # 设置为None以处理所有样本
    }

    # 获取文件路径
    images_path = get_file_paths(config['image_dir'], '*')[:config['sample_limit']]
    masks_path = get_file_paths(config['mask_dir'], '*')[:config['sample_limit']]
    patient_list = [Path(path).name for path in images_path]

    # 初始化特征提取器
    logger.info("初始化特征提取器...")
    extractor = RadiomicsExtractor()
    features = []
    features_name = None

    # 并行处理特征提取
    logger.info(f"开始并行处理特征提取，使用 {config['num_workers']} 个工作进程")
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['num_workers']) as executor:
        args = zip(images_path, masks_path, patient_list)
        results = list(executor.map(extractor.extract_features, args))

        # 处理结果
        for patient, feature_name, feature_value in results:
            if not features_name and feature_name:
                features_name = ['Patient'] + feature_name
            if feature_value:  # 只添加成功提取特征的病例
                features.append([patient] + feature_value)
                logger.info(f"病例 {patient} 特征提取完成")
            else:
                logger.warning(f"病例 {patient} 特征提取失败")

    # 保存结果
    if features_name and features:
        logger.info(f"成功提取 {len(features)} 个病例的特征")
        df = pd.DataFrame(features, columns=features_name)
        save_features(df, config['output_path'])
    else:
        logger.warning("未能提取到任何特征，请检查输入数据和参数设置")


if __name__ == "__main__":
    logger.info("开始运行特征提取程序...")
    try:
        main()
        logger.info("程序运行完成")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")