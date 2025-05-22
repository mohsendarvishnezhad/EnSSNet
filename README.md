EnSSNet


EnSSNet: An Advanced Ensemble Self-Supervised Learning Framework with Mini Batch-Graph Convolutional Network and Deep Curriculum Learning for Robust PolSAR Image Classification


This repository contains a comprehensive and advanced framework for the classification of Polarimetric Synthetic Aperture Radar (PolSAR) images using a combination of self-supervised learning, deep curriculum learning, multi-scale spatial context modeling, and an ensemble of SVM classifiers. The pipeline is designed to be efficient in small-sample scenarios and leverages polarimetric decomposition features along with deep representations.



🧠 Key Features

    ✅ Self-Supervised Pretraining using EfficientNet-B0 and a custom Graph Convolutional Network (GCN)
    🎯 Curriculum Learning Strategy that ranks and introduces samples based on difficulty
    🔍 Multi-Scale Patch-Based Spatial Feature Extraction
    ⚙️ Polarimetric Feature Extraction from decomposition-based indices (e.g., Touzi, Yamaguchi)
    🧪 Group-Wise Feature Processing based on Mutual Information among spectral bands
    🧬 Feature Selection via the proposed Forward and Reverse Rollback Feature Selection methods
    🧮 Ensemble SVM Classification with weighted fusion of probabilistic outputs
    📊 Confusion Matrix, Accuracy Metrics, and final classification map generation



🛠 Requirements
Install dependencies via pip: pip install numpy scikit-learn matplotlib torch torchvision


📂 Project Structure

├── train_with_curriculum.py       # Self-supervised training with curriculum logic

├── extract_features.py            # Feature extraction from pretrained networks

├── mutual_info_band_grouping.py   # Band grouping via mutual information

├── feature_selection.py           # RRFS feature selection method

├── svm_ensemble_classifier.py     # SVM training and probabilistic ensemble fusion

├── utils.py                       # Helper functions

├── main_pipeline.py               # Main execution script

├── README.md                      # Project documentation




🚀 How to Run

Prepare your inputs:
    PolSAR: A 3D numpy array of the input PolSAR image (H × W × Bands)

Execute the main script:
    python main_pipeline.py

Results:
  The trained SVM models will output prediction probabilities.
  Probabilities from each group are fused based on classification accuracy.
  The final map is displayed using matplotlib and saved optionally.



📈 Sample Output

    ✅ Overall Accuracy

    ✅ Confusion Matrix

    ✅ Classification Report 

    🎨 Classified Map Displayed with jet Colormap


