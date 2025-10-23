#!/usr/bin/env python3
"""
CNN Model for Protein Classification

This module provides a convolutional neural network architecture optimized
for protein classification tasks using sequence-based features.

The model uses 2D convolutions to capture local patterns in protein feature
vectors and includes batch normalization and dropout for regularization.
The framework is designed to be adaptable for various protein classification
problems such as subcellular localization, function prediction, enzyme
classification, and more.

Author: Naveen Duhan
Date: 2025-01-17
"""

import os
import gc
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
    BatchNormalization, Input, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D, Attention, LayerNormalization, Add
)
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import config


class ProteinLocalizationCNN:
    """
    Convolutional Neural Network for Protein Classification.
    
    This model processes protein sequence features through convolutional layers
    to predict protein classes. While the current configuration is set up for
    subcellular localization (16 classes), it can be easily adapted for other
    protein classification tasks by modifying the class labels in config.py.
    
    Architecture:
        - Conv2D layer (16 filters, kernel 1x2) + BatchNorm + MaxPool + Dropout
        - Conv2D layer (32 filters, kernel 1x4) + BatchNorm + MaxPool + Dropout
        - Flatten
        - Dense layer (512 units) + Dropout
        - Output layer (configurable number of classes, softmax)
    
    Attributes:
        model (Sequential): Keras model instance
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        input_shape (tuple): Shape of input data
    """
    
    def __init__(self, batch_size=32, learning_rate=0.0001):
        """
        Initialize the CNN model.
        
        Args:
            batch_size (int): Number of samples per batch
            learning_rate (float): Learning rate for SGD optimizer
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = 1  # Will be set during training
        self.model = None
        self.input_shape = None
        
        # Training data placeholders
        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    
    def build_model(self, input_width):
        """
        Build the CNN architecture.
        
        Args:
            input_width (int): Width of input feature vector
        
        Returns:
            Sequential: Compiled Keras model
        
        Example:
            >>> model = ProteinLocalizationCNN()
            >>> model.build_model(input_width=400)
        """
        self.input_shape = (1, input_width, 1)  # (height, width, channels)
        
        self.model = Sequential(name='ProteinLocalization_CNN')
        
        # Input layer
        self.model.add(Input(shape=self.input_shape, name='input'))
        
        # First Convolutional Block
        self.model.add(Conv2D(
            filters=config.CONV_FILTERS_1,
            kernel_size=config.KERNEL_SIZE_1,
            activation='relu',
            kernel_regularizer=l2(config.L2_REGULARIZATION),
            use_bias=False,
            kernel_initializer="he_normal",
            name='conv1'
        ))
        self.model.add(BatchNormalization(name='bn1'))
        self.model.add(MaxPooling2D(
            pool_size=config.POOL_SIZE,
            name='pool1'
        ))
        
        # Second Convolutional Block
        self.model.add(Conv2D(
            filters=config.CONV_FILTERS_2,
            kernel_size=config.KERNEL_SIZE_2,
            activation='relu',
            kernel_regularizer=l2(config.L2_REGULARIZATION),
            name='conv2'
        ))
        self.model.add(BatchNormalization(name='bn2'))
        self.model.add(MaxPooling2D(
            pool_size=config.POOL_SIZE,
            name='pool2'
        ))
        self.model.add(Dropout(
            config.DROPOUT_RATE,
            name='dropout1'
        ))
        
        # Fully Connected Layers
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(
            config.DENSE_UNITS,
            activation='relu',
            name='dense1'
        ))
        self.model.add(Dropout(
            config.DROPOUT_RATE,
            name='dropout2'
        ))
        
        # Output Layer
        self.model.add(Dense(
            config.NUM_CLASSES,
            activation='softmax',
            name='output'
        ))
        
        # Compile Model
        optimizer = SGD(
            learning_rate=self.learning_rate,
            momentum=config.SGD_MOMENTUM,
            nesterov=config.SGD_NESTEROV
        )
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        return self.model
    
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        self.model.summary()
        
        # Calculate total parameters
        trainable_params = np.sum([
            np.prod(v.shape) for v in self.model.trainable_weights
        ])
        non_trainable_params = np.sum([
            np.prod(v.shape) for v in self.model.non_trainable_weights
        ])
        
        print(f"\nTotal trainable parameters: {int(trainable_params):,}")
        print(f"Total non-trainable parameters: {int(non_trainable_params):,}")
        print(f"Total parameters: {int(trainable_params + non_trainable_params):,}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def load_train_data(self, train_data, train_labels):
        """
        Load and reshape training data.
        
        Args:
            train_data (np.array): Training features of shape (n_samples, n_features)
            train_labels (np.array): Training labels of shape (n_samples, n_classes)
        
        Example:
            >>> model.load_train_data(X_train, y_train)
        """
        # Reshape: (n_samples, n_features) -> (n_samples, 1, n_features, 1)
        self._train_data = train_data.reshape(
            train_data.shape[0], 1, train_data.shape[1], 1
        )
        self._train_labels = train_labels
        
        print(f"Training data shape: {self._train_data.shape}")
        print(f"Training labels shape: {self._train_labels.shape}")
    
    def load_test_data(self, test_data, test_labels=None):
        """
        Load and reshape test data.
        
        Args:
            test_data (np.array): Test features
            test_labels (np.array, optional): Test labels
        
        Example:
            >>> model.load_test_data(X_test, y_test)
        """
        self._test_data = test_data.reshape(
            test_data.shape[0], 1, test_data.shape[1], 1
        )
        
        if test_labels is not None:
            self._test_labels = test_labels
            print(f"Test data shape: {self._test_data.shape}")
            print(f"Test labels shape: {self._test_labels.shape}")
        else:
            print(f"Test data shape: {self._test_data.shape}")
    
    # ========================================================================
    # TRAINING AND EVALUATION
    # ========================================================================
    
    def train(self, epochs=None, class_weights=None, validation_split=0.0,
              verbose=1, callbacks=None):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            class_weights (dict): Class weights for imbalanced datasets
            validation_split (float): Fraction of training data for validation
            verbose (int): Verbosity mode (0, 1, or 2)
            callbacks (list): List of Keras callbacks
        
        Returns:
            History: Keras training history object
        
        Example:
            >>> history = model.train(epochs=100, class_weights=weights)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self._train_data is None or self._train_labels is None:
            raise ValueError("Training data not loaded. Call load_train_data() first.")
        
        if epochs is not None:
            self.epochs = epochs
        
        print(f"\nTraining model for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        history = self.model.fit(
            self._train_data,
            self._train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            class_weight=class_weights,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks
        )
        
        gc.collect()
        return history
    
    def predict(self, data=None):
        """
        Generate predictions.
        
        Args:
            data (np.array, optional): Data to predict on. If None, uses loaded test data.
        
        Returns:
            np.array: Prediction probabilities
        
        Example:
            >>> predictions = model.predict(X_test)
        """
        if data is None:
            if self._test_data is None:
                raise ValueError("No test data loaded. Call load_test_data() first.")
            data = self._test_data
        
        predictions = self.model.predict(data, batch_size=self.batch_size)
        gc.collect()
        return predictions
    
    def evaluate(self, data=None, labels=None):
        """
        Evaluate model performance.
        
        Args:
            data (np.array, optional): Test data
            labels (np.array, optional): Test labels
        
        Returns:
            tuple: (loss, accuracy)
        
        Example:
            >>> loss, accuracy = model.evaluate(X_test, y_test)
        """
        if data is None:
            if self._test_data is None or self._test_labels is None:
                raise ValueError("No test data loaded. Call load_test_data() first.")
            data = self._test_data
            labels = self._test_labels
        
        results = self.model.evaluate(
            data, labels,
            batch_size=self.batch_size,
            verbose=0
        )
        gc.collect()
        return results
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    
    def save_model(self, filepath):
        """
        Save complete model to file using the new Keras format.
        
        Args:
            filepath (str or Path): Path to save model (.keras extension recommended)
        
        Example:
            >>> model.save_model("models/my_model.keras")
        
        Note:
            Uses the new Keras 3 format (.keras) which is recommended over
            the legacy HDF5 format (.h5) in TensorFlow 2.16+
        """
        filepath = str(filepath)
        # Ensure .keras extension for new format
        if not filepath.endswith('.keras') and not filepath.endswith('.h5'):
            filepath += '.keras'
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model_from_file(self, filepath):
        """
        Load complete model from file.
        
        Args:
            filepath (str or Path): Path to model file (.keras or .h5)
        
        Example:
            >>> model.load_model_from_file("models/my_model.keras")
        
        Note:
            Supports both new Keras format (.keras) and legacy HDF5 format (.h5)
        """
        self.model = load_model(filepath)
        print(f"Model loaded from: {filepath}")
    
    def save_weights(self, filepath):
        """
        Save only model weights.
        
        Args:
            filepath (str or Path): Path to save weights (.weights.h5 extension)
        
        Example:
            >>> model.save_weights("weights/my_weights.weights.h5")
        
        Note:
            Weight files still use HDF5 format (.weights.h5) as this is
            the standard format for weight-only saving in TensorFlow 2.16+
        """
        filepath = str(filepath)
        # Ensure proper extension for weights
        if not filepath.endswith('.weights.h5') and not filepath.endswith('.h5'):
            filepath += '.weights.h5'
        
        self.model.save_weights(filepath, overwrite=True)
        print(f"Weights saved to: {filepath}")
    
    def load_weights(self, filepath):
        """
        Load model weights.
        
        Args:
            filepath (str or Path): Path to weights file (.weights.h5 or .h5)
        
        Example:
            >>> model.load_weights("weights/my_weights.weights.h5")
        """
        self.model.load_weights(filepath)
        print(f"Weights loaded from: {filepath}")
    
    def get_weights(self):
        """Get current model weights."""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Set model weights."""
        self.model.set_weights(weights)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def shuffle_weights(self):
        """
        Randomly permute model weights for weight initialization experiments.
        
        This is useful for testing the importance of learned weights vs
        random initialization.
        """
        weights = self.model.get_weights()
        shuffled_weights = [
            np.random.permutation(w.flat).reshape(w.shape) 
            for w in weights
        ]
        self.model.set_weights(shuffled_weights)
        print("Weights shuffled randomly")
    
    def reset_weights(self):
        """Reset model weights to initial random values."""
        if self.input_shape is not None:
            input_width = self.input_shape[1]
            self.build_model(input_width)
            print("Weights reset to initial values")
        else:
            print("Cannot reset weights: model not built yet")
    
    def get_layer_output(self, layer_name, input_data):
        """
        Get output of a specific layer.
        
        Args:
            layer_name (str): Name of the layer
            input_data (np.array): Input data
        
        Returns:
            np.array: Layer output
        
        Example:
            >>> conv1_output = model.get_layer_output('conv1', X_test[:10])
        """
        from keras.models import Model
        
        intermediate_layer_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        return intermediate_layer_model.predict(input_data)
    
    @staticmethod
    def clear_session():
        """Clear Keras session to free memory."""
        tf.keras.backend.clear_session()
        gc.collect()
        print("Keras session cleared")


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================

class ModelCheckpointCustom(tf.keras.callbacks.Callback):
    """
    Custom callback to save best model based on validation MCC.
    """
    
    def __init__(self, filepath, monitor='val_accuracy', mode='max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = -np.Inf if mode == 'max' else np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if (self.mode == 'max' and current > self.best) or \
           (self.mode == 'min' and current < self.best):
            self.best = current
            self.model.save(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {current:.4f}, "
                  f"saving model to {self.filepath}")


# ============================================================================
# HYBRID CNN MODEL WITH ESM-2
# ============================================================================

class AttentionFusionLayer(tf.keras.layers.Layer):
    """Gated fusion layer for combining traditional and PLM features."""
    
    def __init__(self, hidden_dim: int = 256, **kwargs):
        super(AttentionFusionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        from keras.layers import Dense, LayerNormalization, Multiply, Add
        
        # Project both inputs to same dimension
        self.dense_traditional = Dense(hidden_dim, activation='relu', name='fusion_trad_proj')
        self.dense_plm = Dense(hidden_dim, activation='relu', name='fusion_plm_proj')
        
        # Gating mechanism to learn importance of each feature type
        self.gate_traditional = Dense(hidden_dim, activation='sigmoid', name='fusion_gate_trad')
        self.gate_plm = Dense(hidden_dim, activation='sigmoid', name='fusion_gate_plm')
        
        # Final fusion layers
        self.dense_fusion = Dense(hidden_dim, activation='relu', name='fusion_combined')
        self.layer_norm = LayerNormalization(name='fusion_norm')
        
    def call(self, inputs):
        traditional_features, plm_features = inputs
        
        # Project features to same dimension
        traditional_proj = self.dense_traditional(traditional_features)
        plm_proj = self.dense_plm(plm_features)
        
        # Compute gates (learned importance weights)
        gate_trad = self.gate_traditional(traditional_features)
        gate_plm = self.gate_plm(plm_features)
        
        # Apply gating
        from keras.layers import Multiply, Add
        gated_trad = Multiply()([traditional_proj, gate_trad])
        gated_plm = Multiply()([plm_proj, gate_plm])
        
        # Combine gated features
        combined = Add()([gated_trad, gated_plm])
        combined = self.layer_norm(combined)
        
        # Final fusion
        fused = self.dense_fusion(combined)
        
        return fused
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'hidden_dim': self.hidden_dim})
        return cfg


class HybridProteinCNN:
    """Hybrid CNN model combining traditional features and ESM-2 embeddings."""
    
    def __init__(self, traditional_input_dim, plm_input_dim=1280, 
                 num_classes=None, batch_size=32, learning_rate=0.001,
                 dropout_rate=0.3, l2_reg=1e-4):
        self.traditional_input_dim = traditional_input_dim
        self.plm_input_dim = plm_input_dim
        self.num_classes = num_classes or config.NUM_CLASSES
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def _build_traditional_branch(self, input_layer):
        from keras.layers import Dense, BatchNormalization, Dropout
        x = Dense(512, activation='relu', kernel_regularizer=l2(self.l2_reg))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        return x
    
    def _build_plm_branch(self, input_layer):
        from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
        x = Dense(512, activation='relu', kernel_regularizer=l2(self.l2_reg))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Reshape((512, 1))(x)
        x = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(self.dropout_rate)(x)
        x = GlobalAveragePooling1D()(x)
        return x
    
    def build_model(self):
        from keras.models import Model
        from keras.layers import Input, Dense, BatchNormalization, Dropout
        traditional_input = Input(shape=(self.traditional_input_dim,), name='traditional_features')
        plm_input = Input(shape=(self.plm_input_dim,), name='plm_features')
        traditional_branch = self._build_traditional_branch(traditional_input)
        plm_branch = self._build_plm_branch(plm_input)
        fusion_layer = AttentionFusionLayer(hidden_dim=256)
        fused_features = fusion_layer([traditional_branch, plm_branch])
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg))(fused_features)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        output = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        self.model = Model(inputs=[traditional_input, plm_input], outputs=output, name='HybridProteinCNN')
        return self.model
    
    def compile_model(self, class_weights=None):
        if self.model is None:
            self.build_model()
        from keras.optimizers import Adam
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Hybrid CNN compiled - Parameters: {self.model.count_params():,}")
        return self.model
    
    def train(self, traditional_features, plm_features, labels, 
              validation_data, epochs=100, class_weights=None, callbacks=None):
        if self.model is None:
            self.compile_model(class_weights)
        val_traditional, val_plm, val_labels = validation_data
        history = self.model.fit(
            [traditional_features, plm_features], labels,
            validation_data=([val_traditional, val_plm], val_labels),
            epochs=epochs, batch_size=self.batch_size,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )
        return history
    
    def predict(self, traditional_features, plm_features):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model.predict([traditional_features, plm_features], batch_size=self.batch_size)
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.save(filepath)
        print(f"Hybrid model saved to {filepath}")
    
    def load_model_from_file(self, filepath):
        from keras.models import load_model
        self.model = load_model(filepath, custom_objects={'AttentionFusionLayer': AttentionFusionLayer})
        print(f"Hybrid model loaded from {filepath}")
