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
    from transformers import EsmModel, EsmTokenizer
    import torch
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: ESM-2 not available. Install with: pip install transformers torch")

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
        """Load ESM-2 model and tokenizer using Hugging Face transformers."""
        from transformers import EsmModel, EsmTokenizer
        import torch
        
        # Map simplified names to Hugging Face model names
        model_mapping = {
            "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
            "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D", 
            "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
            "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
        }
        
        # Allow both formats (with or without 'facebook/' prefix)
        if self.model_name.startswith('facebook/'):
            hf_model_name = self.model_name
        elif self.model_name in model_mapping:
            hf_model_name = model_mapping[self.model_name]
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(model_mapping.keys())}")
        
        try:
            # Load model and tokenizer via Hugging Face
            self.model = EsmModel.from_pretrained(hf_model_name)
            self.tokenizer = EsmTokenizer.from_pretrained(hf_model_name)
            
            self.model.eval()
            self.model.to(self.device)
            
            print(f"ESM-2 model loaded successfully on {self.device}")
            print(f"Model: {hf_model_name}")
            
        except Exception as e:
            print(f"Error loading ESM-2 model on {self.device}: {e}")
            if self.device != "cpu":
                print("Falling back to CPU...")
                self.device = "cpu"
                self.model = EsmModel.from_pretrained(hf_model_name)
                self.tokenizer = EsmTokenizer.from_pretrained(hf_model_name)
                self.model.eval()
                self.model.to(self.device)
                print(f"ESM-2 model loaded successfully on CPU")
            else:
                raise
    
    def extract_embeddings(self, sequences: List[str], 
                          pooling_strategy: str = "mean",
                          max_length: int = 1024) -> np.ndarray:
        """
        Extract ESM-2 embeddings for protein sequences using Hugging Face transformers.
        
        Args:
            sequences (List[str]): List of protein sequences
            pooling_strategy (str): How to pool sequence embeddings
                - 'mean': Average over sequence length
                - 'cls': Use [CLS] token
                - 'max': Max pooling
                - 'last': Last token
            max_length (int): Maximum sequence length (truncate if longer)
        
        Returns:
            np.ndarray: Shape (n_sequences, embedding_dim)
        """
        if not sequences:
            return np.array([])
        
        import torch
        
        # Truncate sequences if too long
        sequences = [seq[:max_length] for seq in sequences]
        
        # Tokenize sequences
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state
            embeddings = outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_dim)
        
        # Apply pooling strategy
        attention_mask = inputs['attention_mask']
        
        if pooling_strategy == "mean":
            # Mean pooling (excluding padding tokens)
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
        elif pooling_strategy == "cls":
            # Use [CLS] token (first token)
            pooled_embeddings = embeddings[:, 0, :]
        elif pooling_strategy == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings[mask_expanded == 0] = -1e9  # Set padding to large negative
            pooled_embeddings = torch.max(embeddings, 1)[0]
        elif pooling_strategy == "last":
            # Use last non-padding token
            sequence_lengths = attention_mask.sum(1) - 1
            pooled_embeddings = embeddings[torch.arange(embeddings.size(0)), sequence_lengths]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return pooled_embeddings.cpu().numpy()
    
    def extract_per_residue_embeddings(self, sequences: List[str], 
                                     max_length: int = 1024) -> List[np.ndarray]:
        """
        Extract per-residue ESM-2 embeddings using Hugging Face transformers.
        
        Args:
            sequences (List[str]): List of protein sequences
            max_length (int): Maximum sequence length
        
        Returns:
            List[np.ndarray]: List of per-residue embeddings
        """
        if not sequences:
            return []
        
        import torch
        
        # Truncate sequences
        sequences = [seq[:max_length] for seq in sequences]
        
        all_embeddings = []
        
        for seq in sequences:
            # Tokenize single sequence
            inputs = self.tokenizer([seq], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_dim)
            
            # Remove padding tokens (keep only actual sequence)
            attention_mask = inputs['attention_mask'][0].cpu().numpy().astype(bool)
            seq_embeddings = embeddings[0].cpu().numpy()[attention_mask]
            
            all_embeddings.append(seq_embeddings)
        
        return all_embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of ESM-2 embeddings."""
        return self.model.config.hidden_size
    
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