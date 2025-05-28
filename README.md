
# 🌌 EnSSNet

**EnSSNet: An Advanced Ensemble Self-Supervised Learning Framework with Mini Batch-Graph Convolutional Network and Deep Curriculum Learning for Robust PolSAR Image Classification**

EnSSNet introduces a robust and modular pipeline for classifying Polarimetric Synthetic Aperture Radar (PolSAR) images. Designed to perform efficiently in small-sample settings, this framework integrates self-supervised learning, graph neural networks, curriculum learning, and ensemble-based classification. It leverages polarimetric decomposition features alongside multi-scale spatial features to achieve high classification accuracy.

---

## 🧠 Key Highlights

- 🔄 **Self-Supervised Feature Learning**  
  Dual-stream pretraining using EfficientNet-B0 and a custom Mini-Batch Graph Convolutional Network (MB-GCN).

- 🎓 **Deep Curriculum Learning (DCL)**  
  Training strategy that introduces samples gradually based on their difficulty score to enhance model convergence.

- 🧩 **Multi-Scale Spatial Context Modeling**  
  Patch-based spatial feature extraction at multiple scales to capture both local and global patterns.

- ⚙️ **Polarimetric Feature Integration**  
  Incorporates features from advanced decomposition methods (e.g., Touzi, Yamaguchi, Cloude, Freeman).

- 🧠 **Mutual Information-Based Band Grouping**  
  Groups spectral bands based on mutual information for structured feature processing.

- 🧬 **Feature Selection via FRFS & RRFS**  
  Implements novel *Forward Rollback Feature Selection* (FRFS) and *Reverse Rollback Feature Selection* (RRFS) for optimal feature subset selection.

- 🤖 **Ensemble SVM Classification**  
  Probabilistic fusion of multiple group-wise SVM classifiers for improved robustness.

- 📊 **Comprehensive Evaluation**  
  Generates classification maps, confusion matrices, and reports including overall accuracy and class-wise metrics.

---

## 📦 Requirements

Install all necessary dependencies using:

```bash
pip install numpy scikit-learn matplotlib torch torchvision
```

---

## 📁 Project Structure

```bash
├── main_pipeline.py              # Main pipeline script
├── train_with_curriculum.py      # Self-supervised training with DCL
├── extract_features.py           # Feature extraction from pretrained networks
├── mutual_info_band_grouping.py  # Band grouping based on mutual information
├── feature_selection.py          # FRFS & RRFS feature selection
├── svm_ensemble_classifier.py    # SVM training and ensemble fusion
├── utils.py                      # Utility functions
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Prepare Input Data
Prepare your input PolSAR image as a 3D NumPy array with the shape:

```python
(Height, Width, Bands)
```

### 2. Run the Main Pipeline
Execute the pipeline using:

```bash
python main_pipeline.py
```

### 3. View Results
- Prediction probabilities from each group are fused based on performance.
- The final classification map is visualized using `matplotlib` (with jet colormap).
- Evaluation metrics including OA, confusion matrix, and report are printed and saved.

---

## 📈 Sample Output

- ✅ **Overall Accuracy (OA)**
- ✅ **Confusion Matrix**
- ✅ **Classification Report**
- ✅ **Final Classified Map** *(visualized with jet colormap)*

---

> ⚡ *EnSSNet* delivers state-of-the-art performance for PolSAR image classification by unifying advanced learning strategies in a streamlined and extensible architecture.

---
