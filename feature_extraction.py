#!/usr/bin/env python3
"""
Protein Sequence Feature Extraction Module

This module provides comprehensive feature extraction methods for protein sequences
designed for various protein classification tasks including:
- Amino Acid Composition (AAC)
- Dipeptide Composition (DPC)
- Tripeptide Composition (TPC)
- Composition-Transition-Distribution (CTD)
- Pseudo Amino Acid Composition (PAAC)
- Quasi-Sequence-Order (QSO)
- Conjoint Triad
- Normalized Moreau-Broto Autocorrelation
- CKSAAP (Composition of k-spaced Amino Acid Pairs)

The framework is adaptable for different protein classification problems such as
subcellular localization, function prediction, enzyme classification, and more.

Author: Naveen Duhan
Date: 2025-01-17
"""

import copy
import math
import numpy as np
from amino_acid_properties import *


class ProteinFeatureExtractor:
    """
    A comprehensive class for extracting numerical features from protein sequences.
    
    Features can be used for machine learning models to predict various protein
    properties such as subcellular localization, function prediction, enzyme
    classification, protein family classification, and more. The framework is
    designed to be adaptable for different protein classification tasks.
    """
    
    def __init__(self):
        """Initialize the feature extractor with amino acid properties."""
        self.amino_acids = AMINO_ACIDS
        self.aa_property_groups = AA_PROPERTY_GROUPS
        self.aa_property_names = AA_PROPERTY_NAMES
        self.conjoint_groups = CONJOINT_GROUPS
        self.grantham_matrix = GRANTHAM_MATRIX
        self.schneider_wrede_matrix = SCHNEIDER_WREDE_MATRIX
        self.aa_to_index = AA_TO_INDEX
        self.aa_property_matrix = AA_PROPERTY_MATRIX_RAW
        self.aa_property_matrix_normalized = AA_PROPERTY_MATRIX_NORMALIZED
        self.aa_index_matrix = AA_INDEX_MATRIX
        self.ctdc_groups = CTDC_GROUPS
        self.ctdc_property_names = CTDC_PROPERTY_NAMES
        self.aa_property_table = AA_PROPERTY_TABLE
        self.eiip_values = EIIP_VALUES
        self.hydrophobicity = HYDROPHOBICITY
        self.hydrophilicity = HYDROPHILICITY
        self.sidechain_volume = SIDECHAIN_VOLUME
    
    # ========================================================================
    # AMINO ACID COMPOSITION FEATURES
    # ========================================================================
    
    def calculate_aac(self, sequence):
        """
        Calculate Amino Acid Composition.
        
        Returns the frequency of each of the 20 standard amino acids in the sequence.
        
        Args:
            sequence (str): Protein sequence
        
        Returns:
            list: Vector of length 20 containing AAC values
        
        Example:
            >>> extractor = ProteinFeatureExtractor()
            >>> features = extractor.calculate_aac("ACDEFGHIKLMNPQRSTVWY")
            >>> len(features)
            20
        """
        seq_length = len(sequence)
        aac_vector = []
        
        for aa in self.amino_acids:
            frequency = round(float(sequence.count(aa)) / seq_length * 100, 2)
            aac_vector.append(frequency)
        
        return aac_vector
    
    def calculate_aac_terminal(self, sequence, n_terminal=30, terminal_type='N'):
        """
        Calculate Amino Acid Composition for terminal regions.
        
        Args:
            sequence (str): Protein sequence
            n_terminal (int): Number of residues from terminal to consider
            terminal_type (str): 'N' for N-terminal, 'C' for C-terminal
        
        Returns:
            list: Vector of length 20 containing terminal AAC values
        """
        if terminal_type == 'N':
            sub_sequence = sequence[:n_terminal]
        else:  # C-terminal
            sub_sequence = sequence[-n_terminal:]
        
        sub_length = len(sub_sequence)
        aac_vector = []
        
        for aa in self.amino_acids:
            frequency = round(float(sub_sequence.count(aa)) / sub_length * 100, 2)
            aac_vector.append(frequency)
        
        return aac_vector
    
    def calculate_aac_enhanced(self, sequence):
        """
        Calculate enhanced AAC with physicochemical properties.
        
        Returns AAC along with physicochemical properties for each amino acid.
        
        Args:
            sequence (str): Protein sequence
        
        Returns:
            list: Enhanced feature vector with AAC and properties
        """
        seq_length = len(sequence)
        enhanced_vector = []
        
        for aa in self.amino_acids:
            count = sequence.count(aa)
            frequency = round(float(count) / seq_length * 100, 2)
            
            # Add count and frequency
            enhanced_vector.append(count)
            enhanced_vector.append(frequency)
            
            # Add physicochemical properties
            for prop in self.aa_property_table[aa]:
                enhanced_vector.append(prop)
            
            # Add hydrophobicity, hydrophilicity, sidechain volume, and EIIP
            enhanced_vector.append(self.hydrophobicity[aa])
            enhanced_vector.append(self.hydrophilicity[aa])
            enhanced_vector.append(self.sidechain_volume[aa])
            enhanced_vector.append(self.eiip_values[aa])
        
        return enhanced_vector
    
    # ========================================================================
    # DIPEPTIDE AND TRIPEPTIDE COMPOSITION
    # ========================================================================
    
    def calculate_dpc(self, sequence):
        """
        Calculate Dipeptide Composition.
        
        Returns the frequency of all possible dipeptides (400 combinations).
        
        Args:
            sequence (str): Protein sequence
        
        Returns:
            list: Vector of length 400 containing DPC values
        """
        seq_length = len(sequence)
        dpc_vector = []
        
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dipeptide = aa1 + aa2
                frequency = round(float(sequence.count(dipeptide)) / (seq_length - 1) * 100, 2)
                dpc_vector.append(frequency)
        
        return dpc_vector
    
    def calculate_dpc_enhanced(self, sequence):
        """
        Calculate enhanced DPC with Schneider-Wrede similarity values.
        
        Args:
            sequence (str): Protein sequence
        
        Returns:
            list: Enhanced DPC vector with similarity values
        """
        seq_length = len(sequence)
        enhanced_vector = []
        
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dipeptide = aa1 + aa2
                count = sequence.count(dipeptide)
                frequency = round(float(count) / (seq_length - 1) * 100, 2)
                
                enhanced_vector.append(frequency)
                enhanced_vector.append(count)
                enhanced_vector.append(self.schneider_wrede_matrix[dipeptide])
        
        return enhanced_vector
    
    def calculate_tpc(self, sequence):
        """
        Calculate Tripeptide Composition.
        
        Returns the frequency of all possible tripeptides (8000 combinations).
        
        Args:
            sequence (str): Protein sequence
        
        Returns:
            list: Vector of length 8000 containing TPC values
        """
        seq_length = len(sequence)
        tpc_vector = []
        
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                for aa3 in self.amino_acids:
                    tripeptide = aa1 + aa2 + aa3
                    frequency = round(float(sequence.count(tripeptide)) / (seq_length - 2) * 100, 2)
                    tpc_vector.append(frequency)
        
        return tpc_vector
    
    # ========================================================================
    # CKSAAP (Composition of K-Spaced Amino Acid Pairs)
    # ========================================================================
    
    def calculate_cksaap(self, sequence, gap=5, order='alphabetically'):
        """
        Calculate CKSAAP features.
        
        Composition of k-spaced amino acid pairs with different gap values.
        
        Args:
            sequence (str): Protein sequence
            gap (int): Maximum gap between amino acid pairs
            order (str): Ordering scheme ('alphabetically', 'polarity', 'sideChainVolume')
        
        Returns:
            list: CKSAAP feature vector
        """
        if gap < 0:
            raise ValueError("Gap should be equal or greater than zero")
        
        if len(sequence) < gap + 2:
            raise ValueError(f"Sequence length should be larger than {gap + 2}")
        
        aa_order = AA_ORDER_SCHEMES[order]
        encodings = []
        
        # Generate all possible amino acid pairs
        aa_pairs = [aa1 + aa2 for aa1 in aa_order for aa2 in aa_order]
        
        for g in range(gap + 1):
            pair_dict = {pair: 0 for pair in aa_pairs}
            total = 0
            
            for i in range(len(sequence)):
                j = i + g + 1
                if i < len(sequence) and j < len(sequence) and \
                   sequence[i] in aa_order and sequence[j] in aa_order:
                    pair_dict[sequence[i] + sequence[j]] += 1
                    total += 1
            
            # Normalize by total count
            for pair in aa_pairs:
                encodings.append(pair_dict[pair] / total if total > 0 else 0)
        
        return encodings
    
    # ========================================================================
    # COMPOSITION-TRANSITION-DISTRIBUTION (CTD)
    # ========================================================================
    
    def _sequence_to_property_string(self, sequence, property_dict):
        """
        Convert amino acid sequence to property string.
        
        Args:
            sequence (str): Protein sequence
            property_dict (dict): Property grouping dictionary
        
        Returns:
            str: Property string with values '1', '2', '3'
        """
        prop_sequence = copy.deepcopy(sequence)
        for group_num, aa_group in property_dict.items():
            for aa in aa_group:
                prop_sequence = prop_sequence.replace(aa, group_num)
        return prop_sequence
    
    def _calculate_composition(self, sequence, property_dict):
        """Calculate composition component of CTD."""
        prop_seq = self._sequence_to_property_string(sequence, property_dict)
        seq_length = len(prop_seq)
        
        composition = []
        for i in ['1', '2', '3']:
            comp = round(float(prop_seq.count(i)) / seq_length, 5)
            composition.append(comp)
        
        return composition
    
    def _calculate_transition(self, sequence, property_dict):
        """Calculate transition component of CTD."""
        prop_seq = self._sequence_to_property_string(sequence, property_dict)
        seq_length = len(prop_seq)
        
        transitions = []
        t12 = round(float(prop_seq.count('12') + prop_seq.count('21')) / (seq_length - 1), 5)
        t13 = round(float(prop_seq.count('13') + prop_seq.count('31')) / (seq_length - 1), 5)
        t23 = round(float(prop_seq.count('23') + prop_seq.count('32')) / (seq_length - 1), 5)
        
        transitions.extend([t12, t13, t23])
        return transitions
    
    def _calculate_distribution(self, sequence, property_dict):
        """Calculate distribution component of CTD."""
        prop_seq = self._sequence_to_property_string(sequence, property_dict)
        seq_length = len(prop_seq)
        distribution = []
        
        for group in ['1', '2', '3']:
            count = prop_seq.count(group)
            positions = []
            
            # Find all positions of this group
            pos = 0
            for _ in range(count):
                pos = prop_seq.find(group, pos) + 1
                if pos > 0:
                    positions.append(pos)
            
            if not positions:
                distribution.extend([0, 0, 0, 0, 0])
            else:
                # Calculate distribution percentiles
                dist = [
                    round(float(positions[0]) / seq_length * 100, 5),
                    round(float(positions[int(math.floor(count * 0.25)) - 1]) / seq_length * 100, 5),
                    round(float(positions[int(math.floor(count * 0.5)) - 1]) / seq_length * 100, 5),
                    round(float(positions[int(math.floor(count * 0.75)) - 1]) / seq_length * 100, 5),
                    round(float(positions[-1]) / seq_length * 100, 5)
                ]
                distribution.extend(dist)
        
        return distribution
    
    def calculate_ctd(self, sequence, components='ctd'):
        """
        Calculate Composition-Transition-Distribution features.
        
        Args:
            sequence (str): Protein sequence
            components (str): Which components to calculate ('c', 't', 'd', or any combination)
        
        Returns:
            list: CTD feature vector (max length 168 for all components)
        """
        components = components.lower()
        result = []
        
        get_composition = 'c' in components
        get_transition = 't' in components
        get_distribution = 'd' in components
        
        for property_dict in self.aa_property_groups:
            if get_composition:
                result.extend(self._calculate_composition(sequence, property_dict))
            if get_transition:
                result.extend(self._calculate_transition(sequence, property_dict))
            if get_distribution:
                result.extend(self._calculate_distribution(sequence, property_dict))
        
        return result
    
    def calculate_ctdc(self, sequence):
        """Calculate composition component only (CTDC)."""
        result = []
        for property_name in self.ctdc_property_names:
            count1 = sum(1 for aa in sequence if aa in CTDC_GROUP1[property_name])
            count2 = sum(1 for aa in sequence if aa in CTDC_GROUP2[property_name])
            
            composition = [
                count1 / len(sequence),
                count2 / len(sequence),
                1 - (count1 + count2) / len(sequence)
            ]
            result.extend(composition)
        
        return result
    
    def calculate_ctdt(self, sequence):
        """Calculate transition component only (CTDT)."""
        result = []
        aa_pairs = [sequence[i:i+2] for i in range(len(sequence) - 1)]
        
        for property_name in self.ctdc_property_names:
            transitions = [0, 0, 0]  # 12/21, 13/31, 23/32
            
            for pair in aa_pairs:
                in_group1 = [pair[i] in CTDC_GROUP1[property_name] for i in range(2)]
                in_group2 = [pair[i] in CTDC_GROUP2[property_name] for i in range(2)]
                in_group3 = [pair[i] in CTDC_GROUP3[property_name] for i in range(2)]
                
                if (in_group1[0] and in_group2[1]) or (in_group2[0] and in_group1[1]):
                    transitions[0] += 1
                elif (in_group1[0] and in_group3[1]) or (in_group3[0] and in_group1[1]):
                    transitions[1] += 1
                elif (in_group2[0] and in_group3[1]) or (in_group3[0] and in_group2[1]):
                    transitions[2] += 1
            
            result.extend([t / len(aa_pairs) for t in transitions])
        
        return result
    
    # ========================================================================
    # CONJOINT TRIAD
    # ========================================================================
    
    def _sequence_to_conjoint_numbers(self, sequence):
        """Convert sequence to conjoint group numbers."""
        aa_to_group = {}
        for group_num, aa_list in self.conjoint_groups.items():
            for aa in aa_list:
                aa_to_group[aa] = str(group_num)
        
        return ''.join([aa_to_group.get(aa, '0') for aa in sequence])
    
    def calculate_conjoint_triad(self, sequence):
        """
        Calculate Conjoint Triad features.
        
        Returns frequency of all possible triad combinations (343 features).
        
        Args:
            sequence (str): Protein sequence
        
        Returns:
            list: Vector of length 343 containing conjoint triad frequencies
        """
        number_sequence = self._sequence_to_conjoint_numbers(sequence)
        result = []
        
        for i in range(1, 8):
            for j in range(1, 8):
                for k in range(1, 8):
                    triad = str(i) + str(j) + str(k)
                    count = number_sequence.count(triad)
                    result.append(count)
        
        return result
    
    def calculate_normalized_conjoint_triad(self, sequence):
        """Calculate normalized conjoint triad features."""
        triad_features = self.calculate_conjoint_triad(sequence)
        max_val = max(triad_features)
        min_val = min(triad_features)
        
        if max_val == min_val:
            return [0.0] * len(triad_features)
        
        normalized = [(val - min_val) / (max_val - min_val) for val in triad_features]
        return normalized
    
    # ========================================================================
    # QUASI-SEQUENCE-ORDER (QSO)
    # ========================================================================
    
    def _calculate_sequence_order_coupling(self, sequence, rank, distance_matrix):
        """Calculate sequence-order-coupling numbers."""
        coupling_numbers = []
        
        for i in range(len(sequence) - rank):
            aa1 = sequence[i]
            aa2 = sequence[i + rank]
            distance = distance_matrix.get(aa1 + aa2, 0)
            coupling_numbers.append(math.pow(distance, 2))
        
        return coupling_numbers
    
    def calculate_qso(self, sequence, max_lag=30, weight=0.1):
        """
        Calculate Quasi-Sequence-Order features.
        
        Args:
            sequence (str): Protein sequence
            max_lag (int): Maximum lag for sequence-order
            weight (float): Weight parameter
        
        Returns:
            list: QSO feature vector
        """
        # Calculate for both Grantham and Schneider-Wrede matrices
        coupling_grantham = []
        coupling_sw = []
        
        for lag in range(1, max_lag + 1):
            coupling_grantham.extend(self._calculate_sequence_order_coupling(
                sequence, lag, self.grantham_matrix))
            coupling_sw.extend(self._calculate_sequence_order_coupling(
                sequence, lag, self.schneider_wrede_matrix))
        
        # Calculate AAC
        aac = self.calculate_aac(sequence)
        
        # Calculate denominators
        denominator_sw = 1 + (weight * sum(coupling_sw[:max_lag]))
        denominator_gr = 1 + (weight * sum(coupling_grantham[:max_lag]))
        
        # Construct QSO features
        qso_features = []
        
        # Normalized AAC with SW
        qso_features.extend([round(aa / denominator_sw, 10) for aa in aac])
        
        # Weighted coupling numbers with SW
        for i in range(max_lag):
            qso_features.append(round(weight * coupling_sw[i] / denominator_sw, 10))
        
        # Normalized AAC with Grantham
        qso_features.extend([round(aa / denominator_gr, 10) for aa in aac])
        
        # Weighted coupling numbers with Grantham
        for i in range(max_lag):
            qso_features.append(round(weight * coupling_grantham[i] / denominator_gr, 10))
        
        return qso_features
    
    # ========================================================================
    # PSEUDO AMINO ACID COMPOSITION (PAAC)
    # ========================================================================
    
    def _calculate_property_distance(self, aa1, aa2):
        """Calculate property distance between two amino acids."""
        idx1 = self.aa_to_index[aa1]
        idx2 = self.aa_to_index[aa2]
        
        distance = sum([
            (self.aa_property_matrix_normalized[i][idx1] - 
             self.aa_property_matrix_normalized[i][idx2]) ** 2 
            for i in range(len(self.aa_property_matrix_normalized))
        ]) / len(self.aa_property_matrix_normalized)
        
        return distance
    
    def calculate_paac(self, sequence, lambda_value=30, weight=0.05):
        """
        Calculate Pseudo Amino Acid Composition.
        
        Args:
            sequence (str): Protein sequence
            lambda_value (int): Lambda parameter for correlation
            weight (float): Weight factor
        
        Returns:
            list: PAAC feature vector
        """
        # Calculate theta values (correlation factors)
        theta = []
        for n in range(1, lambda_value + 1):
            theta_n = sum([
                self._calculate_property_distance(sequence[j], sequence[j + n])
                for j in range(len(sequence) - n)
            ]) / (len(sequence) - n)
            theta.append(theta_n)
        
        # Calculate AAC
        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        
        # Calculate PAAC
        denominator = 1 + weight * sum(theta)
        paac_features = []
        
        # First 20 components (normalized AAC)
        for aa in self.amino_acids:
            paac_features.append(aa_counts[aa] / denominator)
        
        # Lambda components (weighted theta values)
        for theta_val in theta:
            paac_features.append((weight * theta_val) / denominator)
        
        return paac_features
    
    # ========================================================================
    # NORMALIZED MOREAU-BROTO AUTOCORRELATION
    # ========================================================================
    
    def calculate_normalized_moreau_broto(self, sequence, n_lag=30):
        """
        Calculate Normalized Moreau-Broto Autocorrelation.
        
        Args:
            sequence (str): Protein sequence
            n_lag (int): Maximum lag
        
        Returns:
            list: NMBroto feature vector
        """
        # Normalize AAIndex matrix
        aa_idx_normalized = np.copy(self.aa_index_matrix)
        pstd = np.std(aa_idx_normalized, axis=1)
        pmean = np.mean(aa_idx_normalized, axis=1)
        
        for i in range(len(aa_idx_normalized)):
            for j in range(len(aa_idx_normalized[i])):
                aa_idx_normalized[i][j] = (aa_idx_normalized[i][j] - pmean[i]) / pstd[i]
        
        # Calculate autocorrelation
        features = []
        seq_length = len(sequence)
        
        for prop_idx in range(8):  # 8 properties
            for lag in range(1, n_lag + 1):
                if seq_length > n_lag:
                    autocorr = sum([
                        aa_idx_normalized[prop_idx][self.aa_to_index.get(sequence[j], 0)] *
                        aa_idx_normalized[prop_idx][self.aa_to_index.get(sequence[j + lag], 0)]
                        for j in range(seq_length - lag)
                    ]) / (seq_length - lag)
                else:
                    autocorr = 0
                
                features.append(autocorr)
        
        return features
    
    # ========================================================================
    # HYBRID FEATURES
    # ========================================================================
    
    def calculate_hybrid_features(self, sequence, *feature_methods):
        """
        Calculate hybrid features by combining multiple feature extraction methods.
        
        Args:
            sequence (str): Protein sequence
            *feature_methods: Variable number of feature extraction methods
        
        Returns:
            list: Combined feature vector
        
        Example:
            >>> extractor = ProteinFeatureExtractor()
            >>> features = extractor.calculate_hybrid_features(
            ...     "ACDEFGH",
            ...     extractor.calculate_dpc,
            ...     extractor.calculate_aac
            ... )
        """
        combined_features = []
        for method in feature_methods:
            combined_features.extend(method(sequence))
        return combined_features


def get_feature_extractor(feature_name):
    """
    Factory function to get the appropriate feature extraction method.
    
    Args:
        feature_name (str): Name of the feature ('AAC', 'DPC', 'TPC', etc.)
    
    Returns:
        callable: Feature extraction method
    
    Example:
        >>> extractor = ProteinFeatureExtractor()
        >>> method = get_feature_extractor('AAC')
        >>> features = method(extractor, "ACDEFGH")
    """
    extractor = ProteinFeatureExtractor()
    
    feature_map = {
        'AAC': extractor.calculate_aac,
        'DPC': extractor.calculate_dpc,
        'TPC': extractor.calculate_tpc,
        'DPCP': extractor.calculate_dpc_enhanced,
        'CTD': extractor.calculate_ctd,
        'CTDC': extractor.calculate_ctdc,
        'CTDT': extractor.calculate_ctdt,
        'conjoint': extractor.calculate_conjoint_triad,
        'norm_conjoint': extractor.calculate_normalized_conjoint_triad,
        'PAAC': extractor.calculate_paac,
        'QSO': extractor.calculate_qso,
        'NMBroto': extractor.calculate_normalized_moreau_broto,
        'AACN': extractor.calculate_aac_enhanced,
        'CKSAAP': extractor.calculate_cksaap,
    }
    
    return feature_map.get(feature_name, extractor.calculate_aac)
