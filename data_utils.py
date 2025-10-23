#!/usr/bin/env python3
"""
Data Processing Utilities for Protein Classification

This module provides utility functions for protein classification tasks:
- Data preprocessing and sequence cleaning
- Feature generation from protein sequences
- Performance metrics calculation
- Data shuffling and splitting
- Result statistics computation

The framework is designed to be adaptable for various protein classification
problems such as subcellular localization, function prediction, enzyme
classification, and more.

Author: Naveen Duhan
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
import math
import re
from Bio import SeqIO
from feature_extraction import ProteinFeatureExtractor
import config
from plm_feature_extractor import PLMFeatureManager


# ============================================================================
# SEQUENCE PREPROCESSING
# ============================================================================

def clean_sequence(sequence):
    """
    Clean protein sequence by removing invalid amino acid characters.
    
    Args:
        sequence (str): Raw protein sequence
    
    Returns:
        str: Cleaned sequence containing only valid amino acids
    
    Example:
        >>> clean_sequence("ACDEF*GHI-KLM")
        'ACDEFGHIKLM'
    """
    # Keep only valid amino acids
    cleaned = re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', sequence.upper())
    return cleaned


def preprocess_sequences(fasta_file, feature_name, class_label, vector_size):
    """
    Process protein sequences from FASTA file and extract features.
    
    Args:
        fasta_file (str or Path): Path to FASTA file
        feature_name (str): Name of feature to extract ('AAC', 'DPC', etc.)
        class_label (np.array): One-hot encoded class label
        vector_size (int): Expected feature vector size
    
    Returns:
        tuple: (features_array, labels_array, sequence_ids)
               - features_array: np.array of shape (n_sequences, vector_size)
               - labels_array: np.array of shape (n_sequences, n_classes)
               - sequence_ids: list of sequence identifiers
    
    Example:
        >>> features, labels, ids = preprocess_sequences(
        ...     "train_Nucleus.fasta", "DPC", np.array([0,0,1]), 400
        ... )
    """
    extractor = ProteinFeatureExtractor()
    feature_method = get_feature_method(feature_name, extractor)
    
    features_list = []
    labels_list = []
    sequence_ids = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Clean sequence
        cleaned_seq = clean_sequence(str(record.seq))
        
        if len(cleaned_seq) < 10:  # Skip very short sequences
            continue
        
        try:
            # Extract features
            features = feature_method(cleaned_seq)
            features_list.append(features)
            labels_list.append(class_label)
            sequence_ids.append(record.id)
        except Exception as e:
            print(f"Warning: Could not process sequence {record.id}: {e}")
            continue
    
    # Convert to numpy arrays
    features_array = np.zeros((len(features_list), vector_size))
    for i, feat in enumerate(features_list):
        features_array[i, :len(feat)] = feat[:vector_size]
    
    labels_array = np.array(labels_list)
    
    return features_array, labels_array, sequence_ids


def get_feature_method(feature_name, extractor):
    """
    Get the appropriate feature extraction method based on feature name.
    
    Args:
        feature_name (str): Name of the feature
        extractor (ProteinFeatureExtractor): Feature extractor object
    
    Returns:
        callable: Feature extraction method
    """
    feature_methods = {
        'AAC': extractor.calculate_aac,
        'DPC': extractor.calculate_dpc,
        'TPC': extractor.calculate_tpc,
        'DPCP': extractor.calculate_dpc_enhanced,
        'AACN': lambda seq: extractor.calculate_aac_enhanced(seq),
        'AACP': lambda seq: extractor.calculate_aac_enhanced(seq),
        'CTD': extractor.calculate_ctd,
        'CTDC': extractor.calculate_ctdc,
        'CTDT': extractor.calculate_ctdt,
        'norm_conjoint': extractor.calculate_normalized_conjoint_triad,
        'conjoint': extractor.calculate_conjoint_triad,
        'PAAC': extractor.calculate_paac,
        'QSO': extractor.calculate_qso,
        'NMBroto': extractor.calculate_normalized_moreau_broto,
        'CKSAAP': extractor.calculate_cksaap,
        'DPC_AAC': lambda seq: extractor.calculate_hybrid_features(
            seq, extractor.calculate_dpc, extractor.calculate_aac),
        'DPC_NM': lambda seq: extractor.calculate_hybrid_features(
            seq, extractor.calculate_dpc, extractor.calculate_normalized_moreau_broto),
        'DPC_NM_CONJOINT': lambda seq: extractor.calculate_hybrid_features(
            seq, extractor.calculate_dpc, extractor.calculate_conjoint_triad,
            extractor.calculate_normalized_moreau_broto),
        'TPC_DPCP': lambda seq: extractor.calculate_hybrid_features(
            seq, extractor.calculate_dpc_enhanced, extractor.calculate_tpc),
    }
    
    return feature_methods.get(feature_name, extractor.calculate_aac)


# ============================================================================
# DATA MANIPULATION
# ============================================================================

def shuffle_data(features, labels):
    """
    Shuffle features and labels in unison.
    
    Args:
        features (np.array): Feature matrix
        labels (np.array): Label matrix
    
    Returns:
        tuple: (shuffled_features, shuffled_labels)
    
    Example:
        >>> X_shuffled, y_shuffled = shuffle_data(X, y)
    """
    permutation = np.random.permutation(features.shape[0])
    shuffled_features = features[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_features, shuffled_labels


def softmax(z):
    """
    Compute softmax values for each set of scores in z.
    
    Args:
        z (np.array): 2D array of scores
    
    Returns:
        np.array: Softmax probabilities
    """
    assert len(z.shape) == 2, "Input must be 2D array"
    
    # Subtract max for numerical stability
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    
    return e_x / div


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_metrics(confusion_matrix):
    """
    Calculate comprehensive performance metrics from confusion matrix.
    
    Args:
        confusion_matrix (list or np.array): [TN, FP, FN, TP]
    
    Returns:
        list: [TP, FN, TN, FP, Sensitivity, Specificity, Precision, NPV,
               FPR, FDR, FNR, Accuracy, F1-score, MCC]
    
    Example:
        >>> metrics = calculate_metrics([90, 10, 5, 95])
        >>> print(f"Accuracy: {metrics[11]:.3f}")
    """
    tn, fp, fn, tp = confusion_matrix
    
    # Avoid division by zero
    epsilon = 1e-10
    
    # Calculate metrics
    sensitivity = tp / (tp + fn + epsilon)  # Recall, True Positive Rate
    specificity = tn / (tn + fp + epsilon)  # True Negative Rate
    precision = tp / (tp + fp + epsilon)    # Positive Predictive Value
    npv = tn / (tn + fn + epsilon)          # Negative Predictive Value
    fpr = fp / (fp + tn + epsilon)          # False Positive Rate
    fdr = fp / (fp + tp + epsilon)          # False Discovery Rate
    fnr = fn / (fn + tp + epsilon)          # False Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    f1_score = (2 * tp) / (2 * tp + fp + fn + epsilon)
    
    # Matthews Correlation Coefficient
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        mcc = 0
    else:
        mcc = ((tp * tn) - (fp * fn)) / denominator
    
    return [
        tp, fn, tn, fp,
        sensitivity, specificity, precision, npv,
        fpr, fdr, fnr,
        accuracy, f1_score, mcc
    ]


def compute_overall_metrics(class_metrics_df):
    """
    Compute overall weighted metrics from per-class metrics.
    
    Args:
        class_metrics_df (pd.DataFrame): DataFrame with per-class metrics
    
    Returns:
        pd.DataFrame: Single row DataFrame with overall metrics
    
    Example:
        >>> overall = compute_overall_metrics(metrics_df)
    """
    # Calculate weights based on total samples per class
    class_metrics_df['Total'] = class_metrics_df['TP'] + class_metrics_df['FP']
    total_samples = class_metrics_df['Total'].sum()
    
    # Weighted metrics
    weighted_metrics = {}
    metrics_to_weight = ['Sensitivity', 'Specificity', 'Precision', 'NPV',
                         'FPR', 'FDR', 'FNR', 'Accuracy', 'F1score', 'MCC']
    
    for metric in metrics_to_weight:
        weighted_metrics[metric] = (
            (class_metrics_df[metric] * class_metrics_df['Total']).sum() / 
            total_samples
        )
    
    # Sum of confusion matrix elements
    overall = {
        'Metrics': 'Overall',
        'TP': class_metrics_df['TP'].sum(),
        'FP': class_metrics_df['FP'].sum(),
        'TN': class_metrics_df['TN'].sum(),
        'FN': class_metrics_df['FN'].sum(),
    }
    overall.update(weighted_metrics)
    
    return pd.DataFrame([overall])


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_all_training_data(data_dir, feature_name, vector_size, use_plm=True):
    """
    Load all training data for all subcellular locations.
    
    Args:
        data_dir (Path or str): Directory containing training FASTA files
        feature_name (str): Feature extraction method name
        vector_size (int): Feature vector size
    
    Returns:
        tuple: (all_features, all_labels, all_ids)
    
    Example:
        >>> X_train, y_train, train_ids = load_all_training_data(
        ...     config.DATA_DIR, "DPC", 400
        ... )
    """
    all_features = []
    all_labels = []
    all_ids = []
    
    print("\nLoading training data...")
    for i, class_label in enumerate(config.CLASS_LABELS):
        train_file = config.get_data_file(class_label, is_test=False)
        
        if not train_file.exists():
            print(f"Warning: File not found: {train_file}")
            continue
        
        print(f"  Loading {class_label}...")
        one_hot_label = config.get_one_hot_label(class_label)
        
        features, labels, ids = preprocess_sequences(
            train_file, feature_name, np.array(one_hot_label), vector_size
        )
        
        all_features.append(features)
        all_labels.append(labels)
        all_ids.extend(ids)
        print(f"    Loaded {len(features)} sequences")
    
    # Concatenate all data
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal training samples: {len(all_features)}")
    print(f"Feature vector size: {vector_size}")
    
    return all_features, all_labels, all_ids


def load_all_test_data(data_dir, feature_name, vector_size):
    """
    Load all test data for all subcellular locations.
    
    Args:
        data_dir (Path or str): Directory containing test FASTA files
        feature_name (str): Feature extraction method name
        vector_size (int): Feature vector size
    
    Returns:
        tuple: (all_features, all_labels, all_ids)
    
    Example:
        >>> X_test, y_test, test_ids = load_all_test_data(
        ...     config.DATA_DIR, "DPC", 400
        ... )
    """
    all_features = []
    all_labels = []
    all_ids = []
    
    print("\nLoading test data...")
    for class_label in config.CLASS_LABELS:
        test_file = config.get_data_file(class_label, is_test=True)
        
        if not test_file.exists():
            print(f"Warning: File not found: {test_file}")
            continue
        
        print(f"  Loading {class_label}...")
        one_hot_label = config.get_one_hot_label(class_label)
        
        features, labels, ids = preprocess_sequences(
            test_file, feature_name, np.array(one_hot_label), vector_size
        )
        
        all_features.append(features)
        all_labels.append(labels)
        all_ids.extend(ids)
        print(f"    Loaded {len(features)} sequences")
    
    # Concatenate all data
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal test samples: {len(all_features)}")
    
    return all_features, all_labels, all_ids


# ============================================================================
# FASTA CONVERSION UTILITIES
# ============================================================================

def fasta_to_dataframe(fasta_file, class_label):
    """
    Convert FASTA file to pandas DataFrame.
    
    Args:
        fasta_file (str or Path): Path to FASTA file
        class_label (str): Class label for all sequences
    
    Returns:
        pd.DataFrame: DataFrame with columns ['label', 'name', 'sequence']
    
    Example:
        >>> df = fasta_to_dataframe("sequences.fasta", "Nucleus")
    """
    records = []
    sequences = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = clean_sequence(str(record.seq))
        sequences.append(seq)
        records.append(record.id)
    
    df = pd.DataFrame({
        'label': [class_label] * len(records),
        'name': records,
        'sequence': sequences
    })
    
    return df


# ============================================================================
# RESULTS EXPORT UTILITIES
# ============================================================================

def save_predictions_with_probabilities(predictions, probabilities, ids,
                                       output_file, include_class_names=True):
    """
    Save predictions with class probabilities to file.
    
    Args:
        predictions (np.array): Predicted class indices
        probabilities (np.array): Prediction probabilities for all classes
        ids (list): Sequence identifiers
        output_file (Path or str): Output file path
        include_class_names (bool): Whether to include class name column
    
    Example:
        >>> save_predictions_with_probabilities(
        ...     pred, proba, seq_ids, "predictions.txt"
        ... )
    """
    df = pd.DataFrame({'ID': ids})
    
    # Add probability columns
    for i, class_label in enumerate(config.CLASS_LABELS):
        df[f'prob_{class_label}'] = probabilities[:, i]
    
    # Add prediction
    df['predicted_class_idx'] = predictions
    
    if include_class_names:
        df['predicted_class'] = df['predicted_class_idx'].apply(
            config.get_class_name_from_index
        )
    
    # Save to file
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Predictions saved to: {output_file}")


def save_metrics_to_excel(metrics_dict, output_file):
    """
    Save metrics dictionary to Excel file with multiple sheets.
    
    Args:
        metrics_dict (dict): Dictionary with sheet names as keys and DataFrames as values
        output_file (Path or str): Output Excel file path
    
    Example:
        >>> metrics = {'Fold_1': df1, 'Fold_2': df2}
        >>> save_metrics_to_excel(metrics, "metrics.xlsx")
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in metrics_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Metrics saved to: {output_file}")


def load_training_data_with_plm(data_dir, feature_name, vector_size):
    """
    Load training data with both traditional and PLM features.
    
    Args:
        data_dir (Path): Directory containing training data
        feature_name (str): Name of the feature type
        vector_size (int): Size of the feature vector
    
    Returns:
        tuple: (traditional_features, plm_features, labels, protein_ids)
    """
    print(f"Loading training data with PLM features: {feature_name}")
    
    # Load traditional features
    traditional_features, labels, protein_ids = load_all_training_data(
        data_dir, feature_name, vector_size
    )
    
    # Load PLM features
    if config.USE_PLM_FEATURES:
        plm_manager = PLMFeatureManager()
        
        # Load sequences for PLM processing
        sequences = load_sequences_from_fasta(data_dir, feature_name, is_training=True)
        
        # Get or compute PLM embeddings
        plm_features = plm_manager.get_or_compute_embeddings(
            sequences, protein_ids
        )
        
        print(f"PLM features shape: {plm_features.shape}")
    else:
        # Create dummy PLM features if not using PLM
        plm_features = np.zeros((len(protein_ids), config.PLM_EMBEDDING_DIM))
        print("PLM features disabled, using dummy features")
    
    return traditional_features, plm_features, labels, protein_ids


def load_test_data_with_plm(data_dir, feature_name, vector_size):
    """
    Load test data with both traditional and PLM features.
    
    Args:
        data_dir (Path): Directory containing test data
        feature_name (str): Name of the feature type
        vector_size (int): Size of the feature vector
    
    Returns:
        tuple: (traditional_features, plm_features, labels, protein_ids)
    """
    print(f"Loading test data with PLM features: {feature_name}")
    
    # Load traditional features
    traditional_features, labels, protein_ids = load_all_test_data(
        data_dir, feature_name, vector_size
    )
    
    # Load PLM features
    if config.USE_PLM_FEATURES:
        plm_manager = PLMFeatureManager()
        
        # Load sequences for PLM processing
        sequences = load_sequences_from_fasta(data_dir, feature_name, is_training=False)
        
        # Get or compute PLM embeddings
        plm_features = plm_manager.get_or_compute_embeddings(
            sequences, protein_ids
        )
        
        print(f"PLM features shape: {plm_features.shape}")
    else:
        # Create dummy PLM features if not using PLM
        plm_features = np.zeros((len(protein_ids), config.PLM_EMBEDDING_DIM))
        print("PLM features disabled, using dummy features")
    
    return traditional_features, plm_features, labels, protein_ids


def load_sequences_from_fasta(data_dir, feature_name, is_training=True):
    """
    Load protein sequences from FASTA files.
    
    Args:
        data_dir (Path): Directory containing FASTA files
        feature_name (str): Name of the feature type
        is_training (bool): Whether to load training or test sequences
    
    Returns:
        list: List of protein sequences
    """
    sequences = []
    
    if is_training:
        file_pattern = "train_{class_name}.fasta"
    else:
        file_pattern = "test_{class_name}.fasta"
    
    for class_name in config.CLASS_LABELS:
        fasta_file = data_dir / file_pattern.format(class_name=class_name)
        
        if fasta_file.exists():
            with open(fasta_file, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        continue
                    sequences.append(line.strip())
        else:
            print(f"Warning: {fasta_file} not found")
    
    return sequences
