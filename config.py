#!/usr/bin/env python3
"""
Configuration File for Protein Classification using Deep Learning

This module contains all configuration parameters for the deep learning model
including paths, hyperparameters, and class labels. The framework is designed
to be easily adaptable for various protein classification tasks.

Author: Naveen Duhan
Date: 2025-01-17
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "final" / "training"
RAW_DATA_DIR = BASE_DIR / "final" / "all_raw"

# Output directories
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "tmodels"
STATS_DIR = RESULTS_DIR / "tstat"
TEST_STATS_DIR = RESULTS_DIR / "test_stat"
PLOTS_DIR = RESULTS_DIR / "plots"
ROC_DIR = RESULTS_DIR / "ROC"
SLURM_DIR = RESULTS_DIR / "slurm"

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODEL_DIR, STATS_DIR, TEST_STATS_DIR, 
                  PLOTS_DIR, ROC_DIR, SLURM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CLASSIFICATION LABELS
# ============================================================================

# Class labels (16 subcellular locations - can be modified for other tasks)
CLASS_LABELS = [
    'Apical_cell_membrane',
    'Cell_junction',
    'Cell_membrane',
    'Cell_projection',
    'Chromosome',
    'Cytoplasm',
    'Endomembrane_system',
    'Endosome',
    'ER',
    'Golgi',
    'Lysosome',
    'Membrane',
    'Mitochondrion',
    'Nucleus',
    'Peroxisome',
    'Secreted'
]

NUM_CLASSES = len(CLASS_LABELS)

# Class to index mapping
CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# Index to class mapping
IDX_TO_CLASS = {idx: label for idx, label in enumerate(CLASS_LABELS)}

# ============================================================================
# FEATURE CONFIGURATIONS
# ============================================================================

# Available features and their vector sizes
FEATURE_CONFIGS = {
    'AAC': 20,              # Amino Acid Composition
    'DPC': 400,             # Dipeptide Composition
    'TPC': 8000,            # Tripeptide Composition
    'DPCP': 1200,           # Dipeptide Composition with Properties
    'NMBroto': 240,         # Normalized Moreau-Broto Autocorrelation
    'CTD': 168,             # Composition-Transition-Distribution
    'CTDC': 39,             # Composition
    'CTDT': 39,             # Transition
    'conjoint': 343,        # Conjoint Triad
    'PAAC': 50,             # Pseudo Amino Acid Composition
    'QSO': 80,              # Quasi-Sequence-Order
    'CKSAAP': 2400,         # Composition of k-spaced Amino Acid Pairs (gap=5)
    'DPC_AAC': 420,         # Hybrid: DPC + AAC
    'DPC_NM': 640,          # Hybrid: DPC + NMBroto
    'DPC_NM_CONJOINT': 983, # Hybrid: DPC + NMBroto + Conjoint
    'TPC_DPCP': 9200        # Hybrid: TPC + DPCP
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
CROSS_VAL_FOLDS = 5
TEST_SIZE = 0.10
RANDOM_STATE = 424245

# Model architecture parameters
CONV_FILTERS_1 = 16
CONV_FILTERS_2 = 32
KERNEL_SIZE_1 = (1, 2)
KERNEL_SIZE_2 = (1, 4)
POOL_SIZE = (1, 1)
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.0001
DENSE_UNITS = 512

# Optimizer parameters
SGD_MOMENTUM = 0.8
SGD_NESTEROV = True

# ============================================================================
# FILE PATTERNS
# ============================================================================

# Training file patterns
TRAIN_FILE_PATTERN = "train_{}.fasta"
TEST_FILE_PATTERN = "test_{}.fasta"

# Output file patterns
MODEL_FILE_PATTERN = "{}_model_fold_{}.keras"
CLASSIFICATION_REPORT_PATTERN = "{}_training_final_fold_{}.txt"
ROC_DATA_PATTERN = "ROC_{}_training_fold_{}.txt"
METRICS_PATTERN = "MCC_{}_fold_{}.xlsx"
PREDICTION_PATTERN = "{}_predictions_fold_{}.txt"

# Combined output files
COMBINED_TRAIN_METRICS = "MCC_{}_train_all_folds.xlsx"
COMBINED_TEST_METRICS = "MCC_{}_test_all_folds.xlsx"
COMBINED_PREDICTIONS = "{}_predictions_all_folds.xlsx"
COMBINED_ROC_DATA = "ROC_{}_train_all_folds.xlsx"

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

# Metrics to calculate
METRICS_LIST = [
    'TP', 'FP', 'TN', 'FN',
    'Sensitivity', 'Specificity', 'Precision', 'NPV',
    'FPR', 'FDR', 'FNR', 'Accuracy', 'F1score', 'MCC'
]

# ============================================================================
# PROTEIN LANGUAGE MODEL PARAMETERS
# ============================================================================

# ESM-2 model configuration
PLM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # ESM-2 model variant (Hugging Face format)
PLM_EMBEDDING_DIM = 1280  # ESM-2 embedding dimension
PLM_POOLING_STRATEGY = "mean"  # How to pool sequence embeddings
PLM_MAX_LENGTH = 1024  # Maximum sequence length for ESM-2
PLM_CACHE_DIR = MODEL_DIR / "plm_cache"

# Hybrid model configuration
USE_PLM_FEATURES = True  # Enable PLM features
PLM_FEATURE_WEIGHT = 0.5  # Weight for PLM features in fusion
TRADITIONAL_FEATURE_WEIGHT = 0.5  # Weight for traditional features in fusion

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure sizes
FIG_SIZE_SINGLE = (10, 8)
FIG_SIZE_MULTI = (15, 10)
DPI = 300

# Color schemes
COLORS_16 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
]

# Line styles for ROC curves
LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', 
               '-', '--', '-.', ':', '-', '--', '-.', ':']

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FILE = RESULTS_DIR / "training.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# ============================================================================
# SLURM CONFIGURATION (for cluster computing)
# ============================================================================

SLURM_PARTITION = "mahaguru"
SLURM_NODELIST = "chela-g01"
SLURM_GPU = "gpu:tesla:1"
SLURM_TIME = "2-0:00"
SLURM_MEMORY = "32G"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_file(class_label, is_test=False):
    """
    Get the file path for a specific class label.
    
    Args:
        class_label (str): Name of the subcellular location
        is_test (bool): If True, returns test file path, else training file path
    
    Returns:
        Path: Full path to the data file
    """
    pattern = TEST_FILE_PATTERN if is_test else TRAIN_FILE_PATTERN
    return DATA_DIR / pattern.format(class_label)


def get_one_hot_label(class_label):
    """
    Convert class label to one-hot encoded vector.
    
    Args:
        class_label (str): Name of the subcellular location
    
    Returns:
        list: One-hot encoded vector of length NUM_CLASSES
    """
    one_hot = [0] * NUM_CLASSES
    one_hot[CLASS_TO_IDX[class_label]] = 1
    return one_hot


def get_class_name_from_index(index):
    """
    Get class name from index.
    
    Args:
        index (int): Class index (0-15)
    
    Returns:
        str: Class label name
    """
    return IDX_TO_CLASS.get(index, 'Unknown')
