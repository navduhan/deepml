#!/usr/bin/env python3
"""
Visualization Module for Protein Classification Model Performance

This module provides comprehensive visualization tools for evaluating
protein classification model performance including:
- ROC (Receiver Operating Characteristic) curves
- PR (Precision-Recall) curves
- Confusion matrices
- Performance metrics heatmaps
- Training history plots

The framework is designed to work with various protein classification
tasks and can be easily adapted for different class labels and metrics.

Author: Naveen Duhan
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from pathlib import Path
import config


class ModelVisualizer:
    """
    Comprehensive visualization class for protein classification model
    performance evaluation. Designed to work with various protein
    classification tasks and configurable class labels.
    """
    
    def __init__(self, output_dir=None, dpi=300):
        """
        Initialize visualizer.
        
        Args:
            output_dir (Path or str): Directory to save plots
            dpi (int): Resolution for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else config.PLOTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.class_labels = config.CLASS_LABELS
        self.colors = config.COLORS_16
        self.line_styles = config.LINE_STYLES
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
    
    # ========================================================================
    # ROC CURVE VISUALIZATION
    # ========================================================================
    
    def plot_roc_curves_multiclass(self, y_true, y_pred_proba, fold_num=None,
                                    feature_name=None, save=True):
        """
        Plot ROC curves for all classes in a multiclass classification problem.
        
        Args:
            y_true (np.array): True labels (one-hot encoded)
            y_pred_proba (np.array): Predicted probabilities
            fold_num (int): Fold number for cross-validation
            feature_name (str): Name of feature set used
            save (bool): Whether to save the figure
        
        Returns:
            tuple: (figure, roc_auc_dict)
        """
        n_classes = len(self.class_labels)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        fig = plt.figure(figsize=config.FIG_SIZE_MULTI)
        
        # Plot micro-average
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        # Plot macro-average
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=3)
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i],
                    color=self.colors[i % len(self.colors)],
                    linestyle=self.line_styles[i % len(self.line_styles)],
                    linewidth=2,
                    label=f'{self.class_labels[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        
        title = 'ROC Curves - Subcellular Localization Prediction'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save and feature_name:
            filename = f"ROC_curves_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig, roc_auc
    
    def plot_roc_curve_single_class(self, y_true, y_pred_proba, class_idx,
                                    fold_num=None, feature_name=None, save=True):
        """
        Plot ROC curve for a single class.
        
        Args:
            y_true (np.array): True labels (one-hot encoded)
            y_pred_proba (np.array): Predicted probabilities
            class_idx (int): Index of the class to plot
            fold_num (int): Fold number
            feature_name (str): Feature name
            save (bool): Whether to save the figure
        
        Returns:
            tuple: (figure, auc_score)
        """
        fpr, tpr, thresholds = roc_curve(y_true[:, class_idx], y_pred_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        fig = plt.figure(figsize=config.FIG_SIZE_SINGLE)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        
        title = f'ROC Curve - {self.class_labels[class_idx]}'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save and feature_name:
            filename = f"ROC_{self.class_labels[class_idx]}_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig, roc_auc
    
    # ========================================================================
    # PRECISION-RECALL CURVE VISUALIZATION
    # ========================================================================
    
    def plot_pr_curves_multiclass(self, y_true, y_pred_proba, fold_num=None,
                                   feature_name=None, save=True):
        """
        Plot Precision-Recall curves for all classes.
        
        Args:
            y_true (np.array): True labels (one-hot encoded)
            y_pred_proba (np.array): Predicted probabilities
            fold_num (int): Fold number
            feature_name (str): Feature name
            save (bool): Whether to save the figure
        
        Returns:
            tuple: (figure, average_precision_dict)
        """
        n_classes = len(self.class_labels)
        
        # Compute PR curve and Average Precision for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true[:, i], y_pred_proba[:, i])
            average_precision[i] = average_precision_score(
                y_true[:, i], y_pred_proba[:, i])
        
        # Compute micro-average PR curve and Average Precision
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_pred_proba.ravel())
        average_precision["micro"] = average_precision_score(
            y_true, y_pred_proba, average="micro")
        
        # Plot all PR curves
        fig = plt.figure(figsize=config.FIG_SIZE_MULTI)
        
        # Plot micro-average
        plt.plot(recall["micro"], precision["micro"],
                label=f'Micro-average (AP = {average_precision["micro"]:.3f})',
                color='gold', linestyle=':', linewidth=3)
        
        # Plot PR curve for each class
        for i in range(n_classes):
            plt.plot(recall[i], precision[i],
                    color=self.colors[i % len(self.colors)],
                    linestyle=self.line_styles[i % len(self.line_styles)],
                    linewidth=2,
                    label=f'{self.class_labels[i]} (AP = {average_precision[i]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        
        title = 'Precision-Recall Curves - Subcellular Localization Prediction'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save and feature_name:
            filename = f"PR_curves_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig, average_precision
    
    def plot_pr_curve_single_class(self, y_true, y_pred_proba, class_idx,
                                   fold_num=None, feature_name=None, save=True):
        """
        Plot Precision-Recall curve for a single class.
        
        Args:
            y_true (np.array): True labels (one-hot encoded)
            y_pred_proba (np.array): Predicted probabilities
            class_idx (int): Index of the class
            fold_num (int): Fold number
            feature_name (str): Feature name
            save (bool): Whether to save figure
        
        Returns:
            tuple: (figure, average_precision)
        """
        precision, recall, _ = precision_recall_curve(
            y_true[:, class_idx], y_pred_proba[:, class_idx])
        avg_precision = average_precision_score(
            y_true[:, class_idx], y_pred_proba[:, class_idx])
        
        fig = plt.figure(figsize=config.FIG_SIZE_SINGLE)
        
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline
        no_skill = sum(y_true[:, class_idx]) / len(y_true[:, class_idx])
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy',
                label=f'No Skill (AP = {no_skill:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        
        title = f'Precision-Recall Curve - {self.class_labels[class_idx]}'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save and feature_name:
            filename = f"PR_{self.class_labels[class_idx]}_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig, avg_precision
    
    # ========================================================================
    # CONFUSION MATRIX VISUALIZATION
    # ========================================================================
    
    def plot_confusion_matrix(self, y_true, y_pred, fold_num=None,
                              feature_name=None, normalize=False, save=True):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.array): True labels (one-hot encoded or class indices)
            y_pred (np.array): Predicted labels (probabilities or class indices)
            fold_num (int): Fold number
            feature_name (str): Feature name
            normalize (bool): Whether to normalize the confusion matrix
            save (bool): Whether to save the figure
        
        Returns:
            figure: Matplotlib figure object
        """
        # Convert one-hot encoded or probabilities to class indices
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''
        
        fig = plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_labels,
                   yticklabels=self.class_labels,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        title = f'Confusion Matrix{title_suffix}'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save and feature_name:
            norm_str = '_normalized' if normalize else ''
            filename = f"confusion_matrix{norm_str}_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # METRICS VISUALIZATION
    # ========================================================================
    
    def plot_metrics_heatmap(self, metrics_df, fold_num=None,
                             feature_name=None, save=True):
        """
        Plot heatmap of performance metrics across all classes.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with metrics as rows and classes as columns
            fold_num (int): Fold number
            feature_name (str): Feature name
            save (bool): Whether to save the figure
        
        Returns:
            figure: Matplotlib figure object
        """
        # Select numeric metrics only
        numeric_metrics = ['Sensitivity', 'Specificity', 'Precision', 'NPV',
                          'Accuracy', 'F1score', 'MCC']
        
        # Filter metrics
        plot_data = metrics_df[metrics_df['Metrics'].isin(numeric_metrics)]
        plot_data = plot_data.set_index('Metrics')
        
        fig = plt.figure(figsize=config.FIG_SIZE_MULTI)
        sns.heatmap(plot_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        
        title = 'Performance Metrics Heatmap'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Subcellular Location', fontsize=12, fontweight='bold')
        plt.xlabel('Metric', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save and feature_name:
            filename = f"metrics_heatmap_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_class_performance_comparison(self, metrics_df, metric_name='F1score',
                                         fold_num=None, feature_name=None, save=True):
        """
        Plot bar chart comparing a specific metric across all classes.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with metrics
            metric_name (str): Name of the metric to plot
            fold_num (int): Fold number
            feature_name (str): Feature name
            save (bool): Whether to save the figure
        
        Returns:
            figure: Matplotlib figure object
        """
        # Filter for the specific metric
        metric_row = metrics_df[metrics_df['Metrics'] == metric_name]
        
        if metric_row.empty:
            print(f"Metric '{metric_name}' not found in dataframe")
            return None
        
        # Extract class columns (excluding 'Metrics' and 'Overall')
        class_columns = [col for col in metrics_df.columns 
                        if col not in ['Metrics', 'Overall']]
        
        values = metric_row[class_columns].values[0]
        
        fig = plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(class_columns)), values,
                      color=self.colors[:len(class_columns)])
        
        plt.xlabel('Subcellular Location', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name, fontsize=12, fontweight='bold')
        
        title = f'{metric_name} by Subcellular Location'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_columns)), class_columns, rotation=45, ha='right')
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        if save and feature_name:
            filename = f"{metric_name}_comparison_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # TRAINING HISTORY VISUALIZATION
    # ========================================================================
    
    def plot_training_history(self, history, fold_num=None,
                              feature_name=None, save=True):
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            history: Keras history object or dict with 'loss', 'accuracy', etc.
            fold_num (int): Fold number
            feature_name (str): Feature name
            save (bool): Whether to save the figure
        
        Returns:
            figure: Matplotlib figure object
        """
        if hasattr(history, 'history'):
            history = history.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax1.set_title('Model Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Model Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        title = 'Training History'
        if feature_name:
            title += f' - {feature_name}'
        if fold_num is not None:
            title += f' | Fold {fold_num}'
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save and feature_name:
            filename = f"training_history_{feature_name}"
            if fold_num is not None:
                filename += f"_fold_{fold_num}"
            filename += ".png"
            save_path = config.STATS_DIR / filename
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_average_confusion_matrix(self, all_y_true, all_y_pred, 
                                    feature_name=None, normalize=False, save=True):
        """
        Plot average confusion matrix across all folds.
        
        Args:
            all_y_true (list): List of true labels for each fold
            all_y_pred (list): List of predicted labels for each fold
            feature_name (str): Feature name
            normalize (bool): Whether to normalize the confusion matrix
            save (bool): Whether to save the figure
        
        Returns:
            figure: Matplotlib figure object
        """
        # Combine all predictions and true labels
        combined_y_true = np.concatenate(all_y_true)
        combined_y_pred = np.concatenate(all_y_pred)
        
        cm = confusion_matrix(combined_y_true, combined_y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''
        
        fig = plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_labels,
                   yticklabels=self.class_labels,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        title = f'Average Confusion Matrix{title_suffix}'
        if feature_name:
            title += f'\nFeature: {feature_name}'
        title += f' | All Folds Combined'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save and feature_name:
            norm_str = '_normalized' if normalize else ''
            filename = f"ConfusionMatrix_Average{norm_str}_{feature_name}.png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # CROSS-VALIDATION RESULTS VISUALIZATION
    # ========================================================================
    
    def plot_cv_results(self, cv_metrics_dict, metric_name='F1score', save=True):
        """
        Plot cross-validation results showing variation across folds.
        
        Args:
            cv_metrics_dict (dict): Dictionary with fold numbers as keys and
                                   metrics DataFrames as values
            metric_name (str): Name of metric to visualize
            save (bool): Whether to save the figure
        
        Returns:
            figure: Matplotlib figure object
        """
        fig = plt.figure(figsize=(14, 6))
        
        folds = sorted(cv_metrics_dict.keys())
        class_labels = config.CLASS_LABELS
        
        # Prepare data
        fold_data = []
        for fold in folds:
            metrics_df = cv_metrics_dict[fold]
            # Get per-class metrics (exclude 'Overall' row)
            class_rows = metrics_df[metrics_df['Metrics'] != 'Overall']
            if not class_rows.empty and metric_name in class_rows.columns:
                fold_values = class_rows[metric_name].values
                fold_data.append(fold_values)
            else:
                print(f"Warning: Metric '{metric_name}' not found in fold {fold}")
                continue
        
        if not fold_data:
            print(f"Warning: No data found for metric '{metric_name}' across all folds")
            return None
            
        fold_data = np.array(fold_data)
        
        # Check if we have data
        if fold_data.size == 0:
            print(f"Warning: Empty data array for metric '{metric_name}'")
            return None
        
        # Plot with error bars
        means = fold_data.mean(axis=0)
        stds = fold_data.std(axis=0)
        
        # Ensure we have the right number of classes
        if len(means) != len(class_labels):
            print(f"Warning: Expected {len(class_labels)} classes, got {len(means)} values")
            # Truncate or pad as needed
            if len(means) > len(class_labels):
                means = means[:len(class_labels)]
                stds = stds[:len(class_labels)]
            else:
                # Pad with zeros if we have fewer values
                means = np.pad(means, (0, len(class_labels) - len(means)), 'constant')
                stds = np.pad(stds, (0, len(class_labels) - len(stds)), 'constant')
        
        x = np.arange(len(class_labels))
        plt.bar(x, means, yerr=stds, capsize=5, color=self.colors[:len(class_labels)],
               alpha=0.7, error_kw={'linewidth': 2})
        
        plt.xlabel('Subcellular Location', fontsize=12, fontweight='bold')
        plt.ylabel(f'{metric_name} (Mean ± SD)', fontsize=12, fontweight='bold')
        plt.title(f'{metric_name} Across {len(folds)}-Fold Cross-Validation',
                 fontsize=14, fontweight='bold')
        plt.xticks(x, class_labels, rotation=45, ha='right')
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f"cv_{metric_name}_comparison.png"
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def close_all():
        """Close all open figures to free memory."""
        plt.close('all')
