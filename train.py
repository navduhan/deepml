#!/usr/bin/env python3
"""
Unified Training Script for Protein Classification

Supports three training modes:
  - traditional: Uses only sequence-based features (DPC, AAC, etc.)
  - hybrid: Combines traditional features with ESM-2 embeddings
  - plm_only: Uses only ESM-2 embeddings

Usage:
    python train.py <feature_name> <vector_size> [--mode MODE] [--folds N]

Examples:
    python train.py DPC 400                    # Traditional CNN
    python train.py DPC 400 --mode hybrid      # Hybrid CNN + ESM-2
    python train.py DPC 400 --mode plm_only    # PLM-only

Author: Naveen Duhan
Date: 2025-01-17
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Project modules
import config
from cnn_model import ProteinLocalizationCNN, HybridProteinCNN
from data_utils import (
    load_all_training_data, load_all_test_data,
    load_training_data_with_plm, load_test_data_with_plm,
    shuffle_data, calculate_metrics, compute_overall_metrics,
    save_predictions_with_probabilities, save_metrics_to_excel
)
from visualization import ModelVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Protein Classification Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("feature_name", help="Feature type (DPC, AAC, TPC, etc.)")
    parser.add_argument("vector_size", type=int, help="Feature vector dimension")
    parser.add_argument("--mode", choices=["traditional", "hybrid", "plm_only"], 
                       default="traditional", help="Training mode (default: traditional)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    
    args = parser.parse_args()
    
    # Validate feature
    if args.feature_name not in config.FEATURE_CONFIGS:
        print(f"Error: Unknown feature '{args.feature_name}'")
        print(f"Available features: {', '.join(config.FEATURE_CONFIGS.keys())}")
        sys.exit(1)
    
    return args


def load_data(mode, feature_name, vector_size):
    """Load training and test data based on mode."""
    print(f"\nLoading data (mode: {mode})...")
    
    if mode == "traditional":
        # Load only traditional features
        X, y, protein_ids = load_all_training_data(config.DATA_DIR, feature_name, vector_size)
        X_test, y_test, test_ids = load_all_test_data(config.DATA_DIR, feature_name, vector_size)
        return X, None, y, protein_ids, X_test, None, y_test, test_ids
    
    elif mode in ["hybrid", "plm_only"]:
        # Load both traditional and PLM features
        X_trad, X_plm, y, protein_ids = load_training_data_with_plm(
            config.DATA_DIR, feature_name, vector_size
        )
        X_test_trad, X_test_plm, y_test, test_ids = load_test_data_with_plm(
            config.DATA_DIR, feature_name, vector_size
        )
        
        if mode == "hybrid":
            return X_trad, X_plm, y, protein_ids, X_test_trad, X_test_plm, y_test, test_ids
        else:  # plm_only
            return None, X_plm, y, protein_ids, None, X_test_plm, y_test, test_ids


def create_model(mode, feature_name, vector_size, plm_dim, num_classes, class_weight_dict):
    """Create model based on training mode."""
    print(f"\nCreating {mode} model...")
    
    if mode == "traditional":
        model = ProteinLocalizationCNN(
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE
        )
        model.build_model(input_width=vector_size)
        return model
    
    elif mode == "hybrid":
        model = HybridProteinCNN(
            traditional_input_dim=vector_size,
            plm_input_dim=plm_dim,
            num_classes=num_classes,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            dropout_rate=config.DROPOUT_RATE,
            l2_reg=config.L2_REGULARIZATION
        )
        model.compile_model(class_weights=class_weight_dict)
        return model
    
    elif mode == "plm_only":
        model = ProteinLocalizationCNN(
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE
        )
        model.build_model(input_width=plm_dim)
        return model


def train_fold(model, mode, X_train, X_train_plm, y_train, X_val, X_val_plm, y_val, 
               fold_num, feature_name, class_weight_dict, visualizer):
    """Train model for one fold."""
    print(f"\n--- FOLD {fold_num} ---")
    
    if mode == "traditional":
        model.load_train_data(X_train, y_train)
        model.load_test_data(X_val, y_val)
        history = model.train(
            epochs=config.NUM_EPOCHS,
            class_weights=class_weight_dict,
            validation_split=0.0,
            verbose=1
        )
        val_predictions = model.predict()
        
    elif mode == "hybrid":
        history = model.train(
            traditional_features=X_train,
            plm_features=X_train_plm,
            labels=y_train,
            validation_data=(X_val, X_val_plm, y_val),
            epochs=config.NUM_EPOCHS,
            class_weights=class_weight_dict
        )
        val_predictions = model.predict(X_val, X_val_plm)
        
    elif mode == "plm_only":
        model.load_train_data(X_train_plm, y_train)
        model.load_test_data(X_val_plm, y_val)
        history = model.train(
            epochs=config.NUM_EPOCHS,
            class_weights=class_weight_dict,
            validation_split=0.0,
            verbose=1
        )
        val_predictions = model.predict()
    
    # Calculate metrics
    val_metrics = calculate_metrics(y_val, val_predictions)
    
    # Generate visualizations
    prefix = feature_name if mode == "traditional" else f"{mode}_{feature_name}"
    visualizer.plot_roc_curves_multiclass(y_val, val_predictions, prefix, fold_num)
    visualizer.plot_pr_curves_multiclass(y_val, val_predictions, prefix, fold_num)
    visualizer.plot_confusion_matrix(y_val, val_predictions, prefix, fold_num)
    visualizer.plot_confusion_matrix(y_val, val_predictions, prefix, fold_num, normalize=True)
    
    # Save model
    model_path = config.MODEL_DIR / f"{prefix}_model_fold_{fold_num}.keras"
    model.save_model(str(model_path))
    
    print(f"Fold {fold_num} - Accuracy: {val_metrics['accuracy']:.4f}, MCC: {val_metrics['mcc']:.4f}")
    
    return val_metrics, history


def evaluate_test_set(model, mode, X_test, X_test_plm, y_test, test_ids, feature_name, visualizer):
    """Evaluate model on test set."""
    print(f"\n--- TEST SET EVALUATION ---")
    
    if mode == "traditional":
        model.load_test_data(X_test, y_test)
        test_predictions = model.predict()
    elif mode == "hybrid":
        test_predictions = model.predict(X_test, X_test_plm)
    elif mode == "plm_only":
        model.load_test_data(X_test_plm, y_test)
        test_predictions = model.predict()
    
    # Calculate metrics
    test_metrics = calculate_metrics(y_test, test_predictions)
    
    # Generate visualizations
    prefix = f"test_{feature_name}" if mode == "traditional" else f"test_{mode}_{feature_name}"
    visualizer.plot_roc_curves_multiclass(y_test, test_predictions, prefix, "test")
    visualizer.plot_pr_curves_multiclass(y_test, test_predictions, prefix, "test")
    visualizer.plot_confusion_matrix(y_test, test_predictions, prefix, "test")
    visualizer.plot_confusion_matrix(y_test, test_predictions, prefix, "test", normalize=True)
    
    # Save predictions and metrics
    pred_path = config.TEST_STAT_DIR / f"{prefix}_predictions.txt"
    save_predictions_with_probabilities(test_ids, test_predictions, str(pred_path))
    
    metrics_path = config.TEST_STAT_DIR / f"MCC_{prefix}_metrics.xlsx"
    save_metrics_to_excel(test_metrics, str(metrics_path))
    
    print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, MCC: {test_metrics['mcc']:.4f}")
    
    return test_metrics


def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    print("=" * 80)
    print("PROTEIN CLASSIFICATION - UNIFIED TRAINING")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Feature: {args.feature_name}")
    print(f"Vector Size: {args.vector_size}")
    print(f"CV Folds: {args.folds}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 80)
    
    # Create directories
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.STAT_DIR.mkdir(exist_ok=True)
    config.TEST_STAT_DIR.mkdir(exist_ok=True)
    config.PLOTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    X, X_plm, y, protein_ids, X_test, X_test_plm, y_test, test_ids = load_data(
        args.mode, args.feature_name, args.vector_size
    )
    
    print(f"Training samples: {len(protein_ids)}")
    print(f"Test samples: {len(test_ids)}")
    print(f"Number of classes: {y.shape[1]}")
    
    # Shuffle data
    X, X_plm, y, protein_ids = shuffle_data(X, X_plm, y, protein_ids)
    
    # Calculate class weights
    class_labels = np.argmax(y, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Get PLM dimension if needed
    plm_dim = X_plm.shape[1] if X_plm is not None else None
    
    # Cross-validation
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_val_metrics = []
    
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(y), 1):
        # Split data
        if X is not None:
            X_train, X_val = X[train_idx], X[val_idx]
        else:
            X_train, X_val = None, None
            
        if X_plm is not None:
            X_train_plm, X_val_plm = X_plm[train_idx], X_plm[val_idx]
        else:
            X_train_plm, X_val_plm = None, None
        
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train model
        model = create_model(args.mode, args.feature_name, args.vector_size, plm_dim, 
                           y.shape[1], class_weight_dict)
        
        val_metrics, history = train_fold(
            model, args.mode, X_train, X_train_plm, y_train, X_val, X_val_plm, y_val,
            fold_num, args.feature_name, class_weight_dict, visualizer
        )
        
        all_val_metrics.append(val_metrics)
    
    # Compute and save overall training metrics
    overall_metrics = compute_overall_metrics(all_val_metrics)
    prefix = args.feature_name if args.mode == "traditional" else f"{args.mode}_{args.feature_name}"
    overall_path = config.STAT_DIR / f"MCC_{prefix}_train_all_folds.xlsx"
    save_metrics_to_excel(overall_metrics, str(overall_path))
    
    print(f"\n{'='*80}")
    print("OVERALL TRAINING RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"F1-Score: {overall_metrics['f1_score']:.4f}")
    print(f"MCC: {overall_metrics['mcc']:.4f}")
    
    # Test set evaluation (using last model)
    test_metrics = evaluate_test_set(
        model, args.mode, X_test, X_test_plm, y_test, test_ids, 
        args.feature_name, visualizer
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved in: {config.RESULTS_DIR}")
    print(f"Models saved in: {config.MODEL_DIR}")
    print(f"Plots saved in: {config.PLOTS_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
