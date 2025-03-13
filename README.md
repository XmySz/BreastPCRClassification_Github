# Longitudinal self-supervised deep learning for predicting treatment response in breast cancer treated with neoadjuvant therapy: a multicenter, retrospective study

## Project Overview

Breast cancer is the second most common cancer and the leading cause of cancer-related deaths among women. Neoadjuvant therapy (NAT) has emerged as a critical part of the current systemic treatment of breast cancer. Achieving a pathological complete response (pCR) to NAT is a well-established prognostic marker, often associated with favorable outcomes. Despite its prognostic significance, pCR rates remain suboptimal across patient populations, with only about 20% of patients achieving pCR.

This project develops a novel self-supervised deep learning framework that analyzes longitudinal MRI data through temporal learning methods to predict treatment response in breast cancer patients receiving neoadjuvant therapy. The model can help clinicians conduct early response assessments to adjust treatment strategies, improving pCR rates and overall prognosis.

## Research Summary

**Objective**: To develop a self-supervised deep learning model utilizing longitudinal MRI data for early prediction of pathological complete response (pCR) in breast cancer patients undergoing neoadjuvant therapy.

**Methods**: We constructed a self-supervised temporal deep learning framework called BSTNet that integrates three-dimensional convolutional neural networks (3D CNN) for spatial feature extraction, along with Multi-Head Self-Attention (MHSA) and Long Short-Term Memory (LSTM) networks for temporal information processing. The model employs a two-stage training strategy: initial self-supervised pre-training through temporal permutation, followed by supervised fine-tuning for pCR prediction. The study included three independent cohorts of breast cancer patients: training and internal validation sets from the I-SPY2 trial (Center 1), and external validation sets from Guangdong Provincial People's Hospital (Center 2) and Yunnan Cancer Hospital (Center 3).

**Results**: A total of 1,339 patients (3,928 MRI scans) were included in the final analysis. Our BSTNet model achieved areas under the receiver operating characteristic curve (AUC) of 0.882, 0.857, and 0.854 in internal validation, Center 2, and Center 3, respectively, significantly outperforming clinical models, radiomics models, single-scan deep learning models, and multi-temporal models without self-supervised learning. The model demonstrated high specificity in identifying non-pCR patients, with rates of 95.5%, 84.2%, and 94.9% across the three validation sets.

**Conclusion**: The self-supervised temporal deep learning framework we developed effectively predicts pCR in breast cancer patients undergoing neoadjuvant therapy, showing robust performance across multiple independent cohorts and different molecular subtypes. The framework's ability to adapt to varying imaging intervals and protocols suggests its potential for clinical implementation. Future prospective studies are warranted to validate its clinical utility in guiding personalized treatment decisions.

## Dataset Description

This study included three independent cohorts of breast cancer patients who received neoadjuvant therapy (NAT) followed by surgery:

* **Multicenter Data**:
  * Center 1: I-SPY2 clinical trial (2010-2016)
  * Center 2: Guangdong Provincial People's Hospital (March 2015-September 2022)
  * Center 3: Yunnan Cancer Hospital (2020-2022)

* **Cohort Size**:
  * Total patients: 1,339
  * Total MRI scans: 3,928
  * Training set: 529 patients (34.0% achieved pCR)
  * Internal validation set: 133 patients (33.8% achieved pCR)
  * External validation set Center 2: 381 patients (35.2% achieved pCR)
  * External validation set Center 3: 296 patients (34.1% achieved pCR)

* **Inclusion Criteria**:
  * Pathologically confirmed invasive breast cancer
  * Surgery performed after completion of NAT
  * Available pre-treatment MRI scans and any MRI scans taken after 1-4 cycles of treatment
  * Post-operative pathologic evaluation of treatment response to NAT

* **Exclusion Criteria**:
  * Inadequate quality MRI
  * Incomplete or non-standard NAT treatment or surgery
  * Patients with other concurrent malignancies or receiving other oncologic treatments

* **MRI Sequences**: Dynamic contrast-enhanced MRI (DCE-MRI) at the peak enhancement phase of the tumor

## Deep Learning Model

We developed BSTNet (Breast Self-supervised Temporal Network), a temporal deep learning framework specifically designed to analyze longitudinal medical imaging data and predict treatment response:

* **Model Architecture**:
  * BSTNet consists of three main components:
    1. 3D ResNet18 CNN backbone: for spatial feature extraction
    2. Multi-Head Self-Attention (MHSA): for capturing relationships between different time points
    3. Long Short-Term Memory (LSTM) networks: for processing temporal dependencies

* **Self-supervised Learning Method**:
  * Utilizes temporal permutation to create an expanded training dataset
  * Correct temporal sequences are labeled as positive (1) and altered sequences as negative (0)
  * This approach enables the model to learn temporal relationships without additional manual annotation

* **Feature Extraction**:
  * Uses 3D CNN to extract 256-dimensional feature representations from each temporal MRI scan
  * Focuses on tumor region of interest (ROI) with a 3mm extension margin for computational efficiency
  * 8-head attention mechanism processes these features, maintaining consistent feature dimensionality throughout the pipeline

* **Prediction Mechanism**:
  * Two-stage training approach:
    1. Initial self-supervised learning phase: learns patterns of tumor changes over time
    2. Supervised fine-tuning phase: optimized for pCR prediction
  * Final prediction made through fully connected layers

## Code Structure

```
BreastPCRClassification_Github/
├── DeepLearning/               # DeepLearning model definitions
├──├──Experiment
├──├──Preprocess
├── Radiomics/              	# Radiomics model definitions
├──├──Data
├──├──Experiment
├──├──FeatureExtract
├──├──FeatureSelect
```

## Installation Guide

### Requirements

* Python 3.8+
* CUDA 11.0+ (for GPU acceleration)
* PyTorch 1.9+
* Monai 0.9+
* SimpleITK 2.1+
* Pandas, NumPy, SciPy, Scikit-learn

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/XmySz/BreastPCRClassification_Github.git
cd BreastPCRClassification_Github

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage Guide

### Model Training

```python
# Normal training
python DeepLearning/Experiment/train.py 

# Self-supervised pre-training
python DeepLearning/Experiment/train_with_fine_tuning
```

## Experimental Results

BSTNet model demonstrated robust performance across multiple independent validation cohorts:

* **AUC Values**:
  * Internal validation set: 0.882 [95% CI: 0.809-0.942]
  * Center 2 external validation set: 0.857 [95% CI: 0.817-0.895]
  * Center 3 external validation set: 0.854 [95% CI: 0.799-0.903]

* **Specificity** (identifying non-pCR patients):
  * Internal validation set: 95.5% (correctly identified 84 out of 88 non-pCR patients)
  * Center 2: 84.2% (correctly identified 208 out of 247 non-pCR patients)
  * Center 3: 94.9% (correctly identified 185 out of 195 non-pCR patients)

* **Subgroup Analysis**:
  * Stable performance across different hormone receptor (HR) and human epidermal growth factor receptor 2 (HER2) status subgroups
  * Consistent performance across different treatment courses in Center 2 (AUCs: 0.872, 0.850, 0.841, and 0.893)
  * Best performance in the 4th treatment course in Center 3 (AUC: 0.970)

* **Comparison with Other Methods**:
  * Significantly outperformed clinical models (AUCs of 0.738, 0.705, and 0.733, respectively)
  * Significantly outperformed single-scan deep learning models (AUCs of 0.753, 0.643, and 0.608)
  * Significantly outperformed multi-temporal models without self-supervised learning (AUCs of 0.813, 0.788, and 0.759)
  * Significantly outperformed radiomics models (AUCs of 0.827, 0.603, and 0.692)

* **Model Interpretation**:
  * Grad-CAM activation mapping visualized regions contributing most to predictions
  * pCR and non-pCR cases exhibited different activation patterns: centralized activation patterns in pCR cases versus peripheral activation in non-pCR cases

## Notice:This version is a simplified version and the official version will be announced after its release.

## Citation

If you use this code in your research, please cite our paper:

```
Huang Xu, et al., "Longitudinal self-supervised deep learning for predicting treatment response in breast cancer treated with neoadjuvant therapy: a multicenter, retrospective study", 2025
```

## License

This project is open-sourced under the MIT License. See the LICENSE file for details.

## Acknowledgments

We sincerely thank the following institutions and individuals for their support of this project:

- The I-SPY2 clinical trial team for providing valuable data

- Clinical partners at Guangdong Provincial People's Hospital and Yunnan Cancer Hospital

- All radiologists involved in this study, especially X.H., Z.Y.X., and W.Z. for providing expert segmentation and review

- All patients who participated in this study

  

