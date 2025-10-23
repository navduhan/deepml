# Protein Classification using Deep Learning

A comprehensive deep learning framework for protein classification tasks using both traditional Convolutional Neural Networks (CNN) and modern hybrid approaches combining CNN with Protein Language Models (ESM-2).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Available Features](#available-features)
- [Configuration](#configuration)
- [Results](#results)
- [Visualization](#visualization)
- [Citation](#citation)

## Overview

This project provides a flexible deep learning framework for **protein classification tasks** with two powerful approaches:

### **Traditional CNN Approach**
- Uses sequence-based features (DPC, AAC, etc.) with CNN architecture
- Fast, interpretable, and computationally efficient
- Excellent baseline for comparison

### **Hybrid CNN + ESM-2 Approach** (NEW!)
- Combines traditional features with state-of-the-art protein language models
- Leverages both local patterns (CNN) and global protein understanding (ESM-2)
- Attention-based fusion for intelligent feature combination
- Superior performance with modern deep learning

While the current configuration is set up for protein subcellular localization prediction across 16 cellular compartments, the framework can be easily adapted for various protein classification problems such as:

- **Protein Function Prediction**
- **Protein-Protein Interaction Prediction**
- **Enzyme Classification**
- **Protein Family Classification**
- **Disease-Associated Protein Prediction**
- **Protein Subcellular Localization** (current default)

## Features

### Comprehensive Feature Extraction
- **20+ feature extraction methods** including:
  - Amino Acid Composition (AAC)
  - Dipeptide/Tripeptide Composition (DPC/TPC)
  - Pseudo Amino Acid Composition (PAAC)
  - Composition-Transition-Distribution (CTD)
  - Conjoint Triad
  - Quasi-Sequence-Order (QSO)
  - CKSAAP (Composition of k-spaced Amino Acid Pairs)
  - And many more...

### Advanced CNN Architectures
- **Traditional CNN**: Optimized 2D Convolutional Neural Network
- **Hybrid CNN**: Dual-branch architecture with attention-based fusion
- Batch normalization and dropout for regularization
- Class weight balancing for imbalanced datasets
- SGD optimizer with momentum and Nesterov acceleration

### Protein Language Model Integration
- **ESM-2 Integration**: State-of-the-art protein language model
- **Multiple ESM-2 variants**: From 8M to 650M parameters
- **Intelligent fusion**: Attention mechanism combines traditional and PLM features
- **Memory efficient**: Caching system for embeddings
- **Flexible pooling**: Mean, CLS, max, and last token strategies

### Comprehensive Evaluation
- K-fold cross-validation
- Multiple performance metrics (MCC, F1, Sensitivity, Specificity, etc.)
- ROC and Precision-Recall curves
- Confusion matrices
- Per-class and overall performance analysis

### 📈 Rich Visualizations
- ROC curves (per-class and macro/micro-average)
- Precision-Recall curves
- Confusion matrices (normalized and non-normalized)
- Performance metrics heatmaps
- Training history plots
- Cross-validation comparison charts

## Installation

### Requirements

#### Traditional CNN
- Python 3.8+
- TensorFlow 2.x
- BioPython
- scikit-learn
- pandas, numpy, matplotlib, seaborn

#### Hybrid CNN + ESM-2 (NEW!)
- All traditional requirements +
- PyTorch (for ESM-2)
- fair-esm (ESM-2 models)
- transformers (Hugging Face)

### Setup

1. Clone the repository:
```bash
cd /Users/naveen/Desktop/deepml
```

2. Create a conda environment (recommended):
```bash
conda env create -f env.yaml
conda activate tf_metal
```

Or install dependencies manually:
```bash
pip install tensorflow==2.16.2 biopython scikit-learn pandas numpy matplotlib seaborn
```

3. Verify installation:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Project Structure

```
deepml/
├── config.py                     # Central configuration file
├── amino_acid_properties.py      # Amino acid property database
├── feature_extraction.py         # Feature extraction methods
├── cnn_model.py                  # CNN models (Traditional, Hybrid, PLM-only)
├── plm_feature_extractor.py      # ESM-2 feature extraction
├── data_utils.py                 # Data processing utilities
├── visualization.py              # Visualization tools
├── train.py                      # Unified training script (all modes)
├── slurm_launcher.py             # SLURM cluster job launcher
├── env.yaml                      # Apple Silicon environment
├── deepml.yaml                   # Linux environment with ESM-2
├── requirements.txt              # Pip installation requirements
├── README.md                     # This file
│
├── final/
│   ├── all_raw/                  # Raw FASTA files
│   └── training/                 # Train/test split data
│       ├── train_*.fasta
│       └── test_*.fasta
│
└── results/
    ├── tmodels/                  # Trained models
    ├── tstat/                    # Training statistics
    ├── test_stat/                # Test statistics
    ├── plots/                    # Visualization images
    ├── ROC/                      # ROC/PR curve data files
    └── slurm/                    # SLURM job logs
```

## Usage

### Unified Training Script

One simple command for all training modes:

```bash
python train.py <feature_name> <vector_size> [--mode MODE] [--folds N]
```

**Training Modes:**

1. **Traditional CNN** (default - fastest):
```bash
python train.py DPC 400
```

2. **Hybrid CNN + ESM-2** (best performance):
```bash
python train.py DPC 400 --mode hybrid
```

3. **PLM-only** (ESM-2 embeddings only):
```bash
python train.py DPC 400 --mode plm_only
```

**Arguments:**
- `feature_name`: Feature type (DPC, AAC, TPC, etc.)
- `vector_size`: Feature vector dimension
- `--mode`: Training mode - `traditional`, `hybrid`, `plm_only` (default: traditional)
- `--folds`: Number of CV folds (default: 5)


### Available Features

#### Traditional Features
| Feature | Dimensions | Description |
|---------|------------|-------------|
| AAC | 20 | Amino Acid Composition |
| DPC | 400 | Dipeptide Composition |
| TPC | 8000 | Tripeptide Composition |

#### Hybrid Model Features
| Component | Description | Benefits |
|-----------|-------------|----------|
| **Traditional Features** | DPC, AAC, etc. | Local sequence patterns, fast computation |
| **ESM-2 Embeddings** | 1280-dim vectors | Global protein understanding, evolutionary context |
| **Attention Fusion** | Learned combination | Intelligent feature integration |
| DPCP | 1200 | Dipeptide Composition with Properties |
| CTD | 168 | Composition-Transition-Distribution |
| CTDC | 39 | Composition only |
| CTDT | 39 | Transition only |
| conjoint | 343 | Conjoint Triad |
| norm_conjoint | 343 | Normalized Conjoint Triad |
| PAAC | 50 | Pseudo Amino Acid Composition |
| QSO | 80 | Quasi-Sequence-Order |
| NMBroto | 240 | Normalized Moreau-Broto Autocorrelation |
| CKSAAP | 2400 | Composition of k-spaced Amino Acid Pairs |
| DPC_AAC | 420 | Hybrid: DPC + AAC |
| DPC_NM | 640 | Hybrid: DPC + NMBroto |
| DPC_NM_CONJOINT | 983 | Hybrid: DPC + NMBroto + Conjoint |
| TPC_DPCP | 9200 | Hybrid: TPC + DPCP |

### Example Workflow

```bash
# Train using Dipeptide Composition
python train_model.py DPC 400

# Train using CKSAAP features
python train_model.py CKSAAP 2400

# Train using hybrid features
python train_model.py DPC_NM_CONJOINT 983
```

### SLURM Cluster Usage

For large-scale experiments on SLURM clusters:

```bash
python slurm_launcher.py
```

This will launch jobs for all configured features.

## Examples

### Example 1: Traditional CNN Training
```bash
# Train traditional CNN with DPC features
python train.py DPC 400

# With custom folds
python train.py AAC 20 --folds 10

# Results will be saved in:
# - results/tmodels/DPC_model_fold_*.keras
# - results/tstat/MCC_DPC_train_all_folds.xlsx
# - results/plots/ROC_curves_DPC_fold_*.png
```

### Example 2: Hybrid CNN + ESM-2 Training
```bash
# Train hybrid model with DPC + ESM-2
python train.py DPC 400 --mode hybrid

# Results will be saved in:
# - results/tmodels/hybrid_DPC_model_fold_*.keras
# - results/tstat/hybrid_MCC_DPC_train_all_folds.xlsx
# - results/plots/ROC_curves_hybrid_DPC_fold_*.png
```

### Example 3: PLM-only Training
```bash
# Train using only ESM-2 embeddings
python train.py DPC 400 --mode plm_only

# Results will be saved in:
# - results/tmodels/plm_only_DPC_model_fold_*.keras
# - results/tstat/plm_only_MCC_DPC_train_all_folds.xlsx
```

### Example 4: Comparing Different Modes
```bash
# Train all three modes to compare performance
python train.py DPC 400 --mode traditional
python train.py DPC 400 --mode hybrid
python train.py DPC 400 --mode plm_only

# Results will be saved with mode-specific prefixes:
# - results/tstat/MCC_DPC_train_all_folds.xlsx (traditional)
# - results/tstat/MCC_hybrid_DPC_train_all_folds.xlsx (hybrid)
# - results/tstat/MCC_plm_only_DPC_train_all_folds.xlsx (plm_only)
```

## Configuration

### Adapting for Different Classification Tasks

The framework is designed to be easily adaptable for different protein classification problems. Here's how to modify it:

#### 1. Update Class Labels

Edit `config.py` to define your classification categories:

```python
# Example: Protein Function Classification
CLASS_LABELS = [
    'Enzyme',
    'Receptor', 
    'Transporter',
    'Structural',
    'Regulatory',
    'Defense',
    'Storage',
    'Signal'
]

# Example: Enzyme Classification
CLASS_LABELS = [
    'Oxidoreductase',
    'Transferase',
    'Hydrolase',
    'Lyase',
    'Isomerase',
    'Ligase'
]
```

#### 2. Update Data Structure

Organize your data in the `final/training/` directory:
```
final/training/
├── train_Enzyme.fasta
├── train_Receptor.fasta
├── test_Enzyme.fasta
├── test_Receptor.fasta
└── ...
```

#### 3. Modify Hyperparameters

Adjust model parameters in `config.py`:

```python
# Model hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
CROSS_VAL_FOLDS = 5

# CNN architecture
CONV_FILTERS_1 = 16
CONV_FILTERS_2 = 32
DENSE_UNITS = 512
DROPOUT_RATE = 0.5
```

### Adding New Features

1. Implement feature extraction in `feature_extraction.py`:

```python
def calculate_my_feature(self, sequence):
    """Calculate my custom feature."""
    # Your implementation
    return feature_vector
```

2. Register in `get_feature_method()` in `data_utils.py`

3. Add to `FEATURE_CONFIGS` in `config.py`

## Results

### Output Files

After training, results are saved in `results/`:

#### Traditional CNN Results
1. **Models** (`tmodels/`):
   - `{feature}_model_initial.keras` - Initial trained model
   - `{feature}_model_fold_{N}.keras` - Model for each CV fold

2. **Metrics** (`tstat/` and `test_stat/`):
   - `MCC_{feature}_train_all_folds.xlsx` - Training metrics
   - `MCC_{feature}_test_all_folds.xlsx` - Test metrics
   - Per-fold breakdown with all performance metrics

#### Hybrid CNN Results (NEW!)
1. **Models** (`tmodels/`):
   - `hybrid_{feature}_model_fold_{N}.keras` - Hybrid models
   - ESM-2 embeddings cached in `tmodels/plm_cache/`

2. **Metrics** (`tstat/` and `test_stat/`):
   - `hybrid_MCC_{feature}_train_all_folds.xlsx` - Hybrid training metrics
   - `hybrid_MCC_{feature}_test_all_folds.xlsx` - Hybrid test metrics

3. **Comparison Results** (`plots/comparison/`):
   - `model_comparison_metrics.png` - Performance comparison plots
   - `f1_score_by_class.png` - Per-class F1-score comparison
   - `model_comparison_report_{feature}.txt` - Detailed text report

4. **Predictions** (`test_stat/`):
   - `{feature}_predictions_fold_{N}.txt` - Traditional model predictions
   - `hybrid_{feature}_predictions_fold_{N}.txt` - Hybrid model predictions

5. **Visualizations** (`plots/`):
   - ROC curves (multi-class and per-class)
   - Precision-Recall curves
   - Confusion matrices (per-fold and average)
   - Performance comparison charts
   - Training history plots
   - Files: `ConfusionMatrix_{normalized/}_fold_{N}_{feature}.png`

5. **Data Files** (`ROC/`):
   - `ROC_{feature}_train_all_folds.xlsx` - Raw ROC/PR data for analysis

### Performance Metrics

The system calculates comprehensive metrics:

- **Sensitivity** (Recall, True Positive Rate)
- **Specificity** (True Negative Rate)
- **Precision** (Positive Predictive Value)
- **NPV** (Negative Predictive Value)
- **FPR** (False Positive Rate)
- **FDR** (False Discovery Rate)
- **FNR** (False Negative Rate)
- **Accuracy**
- **F1-score**
- **MCC** (Matthews Correlation Coefficient)

## Performance Expectations

### Traditional CNN
- **Training Time**: ~30-60 minutes (depending on dataset size)
- **Memory Usage**: ~2-4 GB RAM
- **Expected Accuracy**: 80-85% (varies by feature type)
- **Best Features**: DPC, TPC, CKSAAP

### Hybrid CNN + ESM-2
- **Training Time**: ~2-4 hours (ESM-2 inference adds overhead)
- **Memory Usage**: ~8-16 GB RAM (ESM-2 model loading)
- **Expected Accuracy**: 85-95% (significant improvement)
- **Best Performance**: DPC + ESM-2 combination

### Performance Comparison
| Model Type | Accuracy | Training Time | Memory | Interpretability |
|------------|----------|---------------|--------|------------------|
| Traditional CNN | 80-85% | Fast | Low | High |
| Hybrid CNN | 85-95% | Moderate | High | Medium |
| ESM-2 Only | 85-90% | Slow | High | Low |

## Visualization

### Using the Visualization Module

```python
from visualization import ModelVisualizer
import numpy as np

# Initialize visualizer
visualizer = ModelVisualizer(output_dir="plots/")

# Plot ROC curves
fig, roc_auc = visualizer.plot_roc_curves_multiclass(
    y_true, y_pred_proba,
    fold_num=1,
    feature_name="DPC"
)

# Plot PR curves
fig, avg_prec = visualizer.plot_pr_curves_multiclass(
    y_true, y_pred_proba,
    fold_num=1,
    feature_name="DPC"
)

# Plot confusion matrix
fig = visualizer.plot_confusion_matrix(
    y_true_labels, y_pred_labels,
    feature_name="DPC",
    normalize=True
)

# Clean up
visualizer.close_all()
```

### Available Visualizations

1. **ROC Curves**
   - Multi-class with micro/macro averaging
   - Per-class individual curves
   - Area Under Curve (AUC) values

2. **Precision-Recall Curves**
   - Multi-class visualization
   - Per-class curves
   - Average Precision (AP) scores

3. **Confusion Matrices**
   - Raw counts and normalized versions
   - Per-fold confusion matrices (training and test)
   - Average confusion matrix across all folds
   - Heatmap visualization with class labels

4. **Performance Metrics**
   - Heatmaps across classes
   - Bar charts for metric comparison
   - Cross-validation fold comparison

5. **Training History**
   - Loss curves
   - Accuracy curves
   - Training vs validation performance

## 📚 Module Documentation

### Core Modules

1. **config.py**
   - Centralized configuration management
   - Path definitions
   - Hyperparameter settings
   - Class label mappings

2. **amino_acid_properties.py**
   - Comprehensive amino acid property database
   - Physicochemical properties
   - Similarity matrices
   - Property groupings

3. **feature_extraction.py**
   - `ProteinFeatureExtractor` class
   - 20+ feature extraction methods
   - Hybrid feature support
   - Well-documented methods

4. **cnn_model.py**
   - `ProteinLocalizationCNN` class
   - Model architecture
   - Training and evaluation
   - Model persistence

5. **data_utils.py**
   - Data loading and preprocessing
   - Sequence cleaning
   - Metrics calculation
   - Result export utilities

6. **visualization.py**
   - `ModelVisualizer` class
   - ROC and PR curve plotting
   - Confusion matrix visualization
   - Training history plots

## Research & Citation

### Author
**Naveen Duhan**

### Date
January 2025

### Citation
If you use this code in your research, please cite:

```bibtex
@software{duhan2025protclass,
  author = {Duhan, Naveen},
  title = {Protein Classification using Deep Learning: Traditional CNN and Hybrid CNN+ESM-2 Approaches},
  year = {2025},
  url = {https://github.com/navduhan/deepml}
}
```

### Related Work

This framework builds upon several key papers:

- **ESM-2**: [Lin et al., 2023](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2) - "Language models enable zero-shot prediction of the effects of mutations on protein function"
- **Protein Classification**: Various works on CNN-based protein classification
- **Attention Mechanisms**: Transformer-based attention for feature fusion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 License

This project is for academic and research purposes. Please contact the author for commercial use.


## 📧 Contact

For questions or feedback, please contact:
- **Author**: Naveen Duhan
- **Email**: [naveen.duhan@outlook.com]

## Acknowledgments

- BioPython for sequence handling
- TensorFlow/Keras for deep learning framework
- scikit-learn for evaluation metrics

## 📖 References

1. GRANTHAM, R. (1974). Amino acid difference formula to help explain protein evolution. *Science*, 185(4154), 862-864.

2. SCHNEIDER, G., & WREDE, P. (1994). The rational design of amino acid sequences by artificial neural networks and simulated molecular evolution. *Biophysical Journal*, 66(2), 335-344.

3. CHOU, K. C. (2001). Prediction of protein cellular attributes using pseudo‐amino acid composition. *Proteins: Structure, Function, and Bioinformatics*, 43(3), 246-255.

---

**Last Updated**: January 17, 2025

**Version**: 2.0
