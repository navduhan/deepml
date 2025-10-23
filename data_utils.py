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

def shuffle_data(features, plm_features, labels, protein_ids):
    """
    Shuffle features, PLM features, labels, and protein IDs in unison.
    
    Args:
        features (np.array or None): Traditional feature matrix
        plm_features (np.array or None): PLM feature matrix
        labels (np.array): Label matrix
        protein_ids (list): List of protein identifiers
    
    Returns:
        tuple: (shuffled_features, shuffled_plm_features, shuffled_labels, shuffled_ids)
    
    Example:
        >>> X_shuffled, X_plm_shuffled, y_shuffled, ids_shuffled = shuffle_data(X, X_plm, y, ids)
    """
    # Determine size from labels or first non-None features
    if features is not None:
        size = features.shape[0]
    elif plm_features is not None:
        size = plm_features.shape[0]
    else:
        size = labels.shape[0]
    
    # Generate random permutation
    permutation = np.random.permutation(size)
    
    # Shuffle each component
    shuffled_features = features[permutation] if features is not None else None
    shuffled_plm_features = plm_features[permutation] if plm_features is not None else None
    shuffled_labels = labels[permutation]
    shuffled_ids = [protein_ids[i] for i in permutation]
    
    return shuffled_features, shuffled_plm_features, shuffled_labels, shuffled_ids


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

def calculate_metrics(y_true, y_pred_proba):
    """
    Calculate comprehensive performance metrics for multi-class classification.
    
    Args:
        y_true (np.array): True labels (one-hot encoded), shape (n_samples, n_classes)
        y_pred_proba (np.array): Predicted probabilities, shape (n_samples, n_classes)
    
    Returns:
        dict: Dictionary containing overall and per-class metrics
    
    Example:
        >>> metrics = calculate_metrics(y_true, y_pred_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, matthews_corrcoef
    )
    
    # Convert to class labels
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    f1_macro = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
    
    # Per-class metrics
    f1_per_class = f1_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
    precision_per_class = precision_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision_macro,
        'recall': recall_macro,
        'mcc': mcc,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'confusion_matrix': cm,
        'y_true': y_true_labels,
        'y_pred': y_pred_labels
    }


def compute_overall_metrics(all_fold_metrics):
    """
    Compute average metrics across all folds.
    
    Args:
        all_fold_metrics (list): List of metric dictionaries from each fold
    
    Returns:
        dict: Average metrics across all folds
    
    Example:
        >>> overall = compute_overall_metrics([fold1_metrics, fold2_metrics])
    """
    if not all_fold_metrics:
        return {}
    
    # Average scalar metrics across folds
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_fold_metrics]),
        'f1_weighted': np.mean([m['f1_weighted'] for m in all_fold_metrics]),
        'precision': np.mean([m['precision'] for m in all_fold_metrics]),
        'recall': np.mean([m['recall'] for m in all_fold_metrics]),
        'mcc': np.mean([m['mcc'] for m in all_fold_metrics]),
    }
    
    # Average per-class metrics
    num_classes = len(all_fold_metrics[0]['f1_per_class'])
    avg_metrics['f1_per_class'] = np.mean(
        [m['f1_per_class'] for m in all_fold_metrics], axis=0
    )
    avg_metrics['precision_per_class'] = np.mean(
        [m['precision_per_class'] for m in all_fold_metrics], axis=0
    )
    avg_metrics['recall_per_class'] = np.mean(
        [m['recall_per_class'] for m in all_fold_metrics], axis=0
    )
    
    return avg_metrics


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

def save_predictions_with_probabilities(ids, probabilities, output_file, include_class_names=True):
    """
    Save predictions with class probabilities to file.
    
    Args:
        ids (list): Sequence identifiers
        probabilities (np.array): Prediction probabilities for all classes
        output_file (Path or str): Output file path
        include_class_names (bool): Whether to include class name column
    
    Example:
        >>> save_predictions_with_probabilities(seq_ids, pred_proba, "predictions.txt")
    """
    # Get predicted class indices from probabilities
    predictions = np.argmax(probabilities, axis=1)
    
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
    Save metrics dictionary to Excel file.
    
    Args:
        metrics_dict (dict): Dictionary with metric names and values
        output_file (Path or str): Output Excel file path
    
    Example:
        >>> metrics = {'accuracy': 0.95, 'f1_score': 0.93}
        >>> save_metrics_to_excel(metrics, "metrics.xlsx")
    """
    # Convert metrics dict to DataFrame
    overall_metrics = {
        'Metric': ['Accuracy', 'F1-Score (Macro)', 'F1-Score (Weighted)', 
                   'Precision', 'Recall', 'MCC'],
        'Value': [
            metrics_dict.get('accuracy', 0),
            metrics_dict.get('f1_score', 0),
            metrics_dict.get('f1_weighted', 0),
            metrics_dict.get('precision', 0),
            metrics_dict.get('recall', 0),
            metrics_dict.get('mcc', 0)
        ]
    }
    df_overall = pd.DataFrame(overall_metrics)
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_overall.to_excel(writer, sheet_name='Overall', index=False)
        
        # Add per-class metrics if available
        if 'f1_per_class' in metrics_dict:
            per_class_metrics = {
                'Class': list(range(len(metrics_dict['f1_per_class']))),
                'F1-Score': metrics_dict['f1_per_class'],
                'Precision': metrics_dict['precision_per_class'],
                'Recall': metrics_dict['recall_per_class']
            }
            df_per_class = pd.DataFrame(per_class_metrics)
            df_per_class.to_excel(writer, sheet_name='Per-Class', index=False)
    
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
    import time
    print(f"Loading training data with PLM features: {feature_name}")
    
    # Load traditional features
    start_time = time.time()
    traditional_features, labels, protein_ids = load_all_training_data(
        data_dir, feature_name, vector_size
    )
    trad_time = time.time() - start_time
    print(f"Traditional feature extraction time: {trad_time:.1f}s")
    
    # Load PLM features
    if config.USE_PLM_FEATURES:
        plm_manager = PLMFeatureManager()
        
        # Load sequences for PLM processing
        sequences = load_sequences_from_fasta(data_dir, feature_name, is_training=True)
        
        # Get or compute PLM embeddings
        plm_features = plm_manager.get_or_compute_embeddings(
            sequences, protein_ids, batch_size=config.PLM_BATCH_SIZE
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
    import time
    print(f"Loading test data with PLM features: {feature_name}")
    
    # Load traditional features
    start_time = time.time()
    traditional_features, labels, protein_ids = load_all_test_data(
        data_dir, feature_name, vector_size
    )
    trad_time = time.time() - start_time
    print(f"Traditional feature extraction time: {trad_time:.1f}s")
    
    # Load PLM features
    if config.USE_PLM_FEATURES:
        plm_manager = PLMFeatureManager()
        
        # Load sequences for PLM processing
        sequences = load_sequences_from_fasta(data_dir, feature_name, is_training=False)
        
        # Get or compute PLM embeddings
        plm_features = plm_manager.get_or_compute_embeddings(
            sequences, protein_ids, batch_size=config.PLM_BATCH_SIZE
        )
        
        print(f"PLM features shape: {plm_features.shape}")
    else:
        # Create dummy PLM features if not using PLM
        plm_features = np.zeros((len(protein_ids), config.PLM_EMBEDDING_DIM))
        print("PLM features disabled, using dummy features")
    
    return traditional_features, plm_features, labels, protein_ids


def load_sequences_from_fasta(data_dir, feature_name, is_training=True):
    """
    Load protein sequences from FASTA files using BioPython.
    
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
            # Use BioPython to properly parse FASTA format
            for record in SeqIO.parse(fasta_file, "fasta"):
                seq = clean_sequence(str(record.seq))
                if seq:  # Only add non-empty sequences
                    sequences.append(seq)
        else:
            print(f"Warning: {fasta_file} not found")
    
    print(f"Loaded {len(sequences)} protein sequences from FASTA files")
    return sequences
