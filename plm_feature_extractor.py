#!/usr/bin/env python3
"""
Protein Language Model Feature Extractor

This module provides ESM-2 integration for extracting protein embeddings
that capture long-range dependencies and complex protein patterns.

Author: Naveen Duhan
Date: 2025-01-17
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings("ignore")

try:
    import esm
    from transformers import EsmModel, EsmTokenizer
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: ESM-2 not available. Install with: pip install fair-esm transformers torch")

import config


class ESM2FeatureExtractor:
    """
    ESM-2 feature extractor for protein sequences.
    
    ESM-2 (Evolutionary Scale Modeling) is a protein language model
    that learns protein representations from evolutionary data.
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", 
                 device: str = "auto", cache_dir: Optional[str] = None):
        """
        Initialize ESM-2 feature extractor.
        
        Args:
            model_name (str): ESM-2 model variant
                - esm2_t33_650M_UR50D: 650M parameters (recommended)
                - esm2_t12_35M_UR50D: 35M parameters (faster)
                - esm2_t6_8M_UR50D: 8M parameters (lightweight)
            device (str): Device to run model on ('auto', 'cpu', 'cuda')
            cache_dir (str): Directory to cache model files
        """
        if not ESM_AVAILABLE:
            raise ImportError("ESM-2 not available. Install with: pip install fair-esm transformers torch")
        
        self.model_name = model_name
        self.cache_dir = cache_dir or str(config.MODEL_DIR / "esm_cache")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading ESM-2 model: {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load ESM-2 model and tokenizer."""
        try:
            # Map model names to ESM-2 loading functions
            model_funcs = {
                "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
                "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
                "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
                "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
            }
            
            if self.model_name not in model_funcs:
                raise ValueError(f"Unknown model: {self.model_name}. Available: {list(model_funcs.keys())}")
            
            # Load model and alphabet
            self.model, self.alphabet = model_funcs[self.model_name]()
            self.model.eval()
            self.model.to(self.device)
            
            # Get batch converter
            self.batch_converter = self.alphabet.get_batch_converter()
            
            print(f"ESM-2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading ESM-2 model: {e}")
            if self.device != "cpu":
                print("Falling back to CPU...")
                self.device = "cpu"
                self.model, self.alphabet = model_funcs[self.model_name]()
                self.model.eval()
                self.model.to(self.device)
                self.batch_converter = self.alphabet.get_batch_converter()
            else:
                raise
    
    def extract_embeddings(self, sequences: List[str], 
                          pooling_strategy: str = "mean",
                          max_length: int = 1024) -> np.ndarray:
        """
        Extract ESM-2 embeddings for protein sequences.
        
        Args:
            sequences (List[str]): List of protein sequences
            pooling_strategy (str): How to pool sequence embeddings
                - 'mean': Average over sequence length
                - 'cls': Use [CLS] token (if available)
                - 'max': Max pooling
                - 'last': Last token
            max_length (int): Maximum sequence length (truncate if longer)
        
        Returns:
            np.ndarray: Shape (n_sequences, embedding_dim)
        """
        if not sequences:
            return np.array([])
        
        # Truncate sequences if too long
        sequences = [seq[:max_length] for seq in sequences]
        
        # Prepare batch data
        batch_data = [(i, seq) for i, seq in enumerate(sequences)]
        
        # Tokenize and convert to batch
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])  # Use last layer
            embeddings = results["representations"][33]  # Shape: (batch, seq_len, hidden_dim)
        
        # Apply pooling strategy
        if pooling_strategy == "mean":
            # Mask out padding tokens
            attention_mask = (batch_tokens != self.alphabet.padding_idx).float()
            pooled_embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        elif pooling_strategy == "cls":
            # Use first token (usually [CLS])
            pooled_embeddings = embeddings[:, 0, :]
        elif pooling_strategy == "max":
            # Max pooling
            attention_mask = (batch_tokens != self.alphabet.padding_idx).float()
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            pooled_embeddings = masked_embeddings.max(1)[0]
        elif pooling_strategy == "last":
            # Use last non-padding token
            attention_mask = (batch_tokens != self.alphabet.padding_idx).long()
            last_indices = attention_mask.sum(1) - 1
            pooled_embeddings = embeddings[torch.arange(embeddings.size(0)), last_indices]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return pooled_embeddings.cpu().numpy()
    
    def extract_per_residue_embeddings(self, sequences: List[str], 
                                     max_length: int = 1024) -> List[np.ndarray]:
        """
        Extract per-residue ESM-2 embeddings.
        
        Args:
            sequences (List[str]): List of protein sequences
            max_length (int): Maximum sequence length
        
        Returns:
            List[np.ndarray]: List of per-residue embeddings
        """
        if not sequences:
            return []
        
        # Truncate sequences
        sequences = [seq[:max_length] for seq in sequences]
        
        all_embeddings = []
        
        for seq in sequences:
            # Prepare single sequence
            batch_data = [(0, seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])
                embeddings = results["representations"][33]  # Shape: (1, seq_len, hidden_dim)
            
            # Remove padding and special tokens
            attention_mask = (batch_tokens[0] != self.alphabet.padding_idx).cpu().numpy()
            seq_embeddings = embeddings[0].cpu().numpy()[attention_mask]
            
            all_embeddings.append(seq_embeddings)
        
        return all_embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of ESM-2 embeddings."""
        return self.model.embed_tokens.embedding_dim
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       sequence_ids: List[str], 
                       output_file: Union[str, Path]):
        """Save embeddings to file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_file,
            embeddings=embeddings,
            sequence_ids=sequence_ids,
            model_name=self.model_name
        )
        print(f"Embeddings saved to {output_file}")
    
    def load_embeddings(self, input_file: Union[str, Path]) -> tuple:
        """Load embeddings from file."""
        data = np.load(input_file)
        return data['embeddings'], data['sequence_ids'].tolist()


class PLMFeatureManager:
    """
    Manager class for handling PLM features in the training pipeline.
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        """Initialize PLM feature manager."""
        self.extractor = ESM2FeatureExtractor(model_name)
        self.embedding_dim = self.extractor.get_embedding_dim()
        self.cache_dir = Path(config.MODEL_DIR) / "plm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_or_compute_embeddings(self, sequences: List[str], 
                                 sequence_ids: List[str],
                                 force_recompute: bool = False) -> np.ndarray:
        """
        Get embeddings from cache or compute them.
        
        Args:
            sequences (List[str]): Protein sequences
            sequence_ids (List[str]): Unique identifiers for sequences
            force_recompute (bool): Force recomputation even if cached
        
        Returns:
            np.ndarray: ESM-2 embeddings
        """
        # Create cache file name
        cache_file = self.cache_dir / f"embeddings_{self.extractor.model_name}.npz"
        
        # Load from cache if available and not forcing recompute
        if cache_file.exists() and not force_recompute:
            try:
                print("Loading ESM-2 embeddings from cache...")
                embeddings, cached_ids = self.extractor.load_embeddings(cache_file)
                
                # Check if all sequences are cached
                if set(sequence_ids).issubset(set(cached_ids)):
                    # Get indices for requested sequences
                    id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}
                    indices = [id_to_idx[sid] for sid in sequence_ids]
                    return embeddings[indices]
                else:
                    print("Cache incomplete, recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing...")
        
        # Compute embeddings
        print("Computing ESM-2 embeddings...")
        embeddings = self.extractor.extract_embeddings(sequences)
        
        # Save to cache
        self.extractor.save_embeddings(embeddings, sequence_ids, cache_file)
        
        return embeddings