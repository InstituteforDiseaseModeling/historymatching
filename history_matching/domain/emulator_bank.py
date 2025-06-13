"""
EmulatorBank domain object for history matching.
"""

from typing import Dict, List, Optional, Union
import os
import pickle
import copy
import logging

from ..emulators.base import BaseEmulator

logger = logging.getLogger(__name__)


class EmulatorBank:
    """
    Manages storage and retrieval of emulators across iterations.
    
    The EmulatorBank stores emulators organized by iteration number and
    feature name, providing methods for adding, retrieving, and managing
    the emulator collection.
    """
    
    def __init__(self):
        """Initialize empty emulator bank."""
        self._emulators: Dict[int, Dict[str, BaseEmulator]] = {}
        
    def add_emulator(self, iteration: int, feature: str, emulator: BaseEmulator):
        """
        Add an emulator for a specific iteration and feature.
        
        Args:
            iteration: Iteration number (1-based)
            feature: Feature name
            emulator: Trained emulator instance
        """
        if not isinstance(emulator, BaseEmulator):
            raise TypeError(f"Expected BaseEmulator instance, got {type(emulator)}")
            
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")
            
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError(f"Feature must be a non-empty string, got {feature}")
            
        # Initialize iteration dict if needed
        if iteration not in self._emulators:
            self._emulators[iteration] = {}
            
        self._emulators[iteration][feature] = emulator
        
    def get_emulator(self, iteration: int, feature: str) -> Optional[BaseEmulator]:
        """
        Retrieve a specific emulator.
        
        Args:
            iteration: Iteration number
            feature: Feature name
            
        Returns:
            Emulator instance or None if not found
        """
        if iteration not in self._emulators:
            return None
            
        return self._emulators[iteration].get(feature)
        
    def get_emulators_for_iteration(self, iteration: int) -> Dict[str, BaseEmulator]:
        """
        Get all emulators for a specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Dict mapping feature names to emulators
        """
        return self._emulators.get(iteration, {}).copy()
        
    def get_latest_emulators(self) -> Dict[str, BaseEmulator]:
        """
        Get emulators from the most recent iteration.
        
        Returns:
            Dict mapping feature names to emulators from latest iteration
        """
        if not self._emulators:
            return {}
            
        latest_iteration = max(self._emulators.keys())
        return self.get_emulators_for_iteration(latest_iteration)
        
    def get_all_iterations(self) -> List[int]:
        """
        Get list of iteration numbers with emulators.
        
        Returns:
            Sorted list of iteration numbers
        """
        return sorted(self._emulators.keys())
        
    def get_features_for_iteration(self, iteration: int) -> List[str]:
        """
        Get feature names for a specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            List of feature names
        """
        if iteration not in self._emulators:
            return []
            
        return list(self._emulators[iteration].keys())
        
    def get_all_features(self) -> List[str]:
        """
        Get all unique feature names across all iterations.
        
        Returns:
            List of unique feature names
        """
        all_features = set()
        for iteration_emulators in self._emulators.values():
            all_features.update(iteration_emulators.keys())
        return sorted(list(all_features))
        
    def has_emulator(self, iteration: int, feature: str) -> bool:
        """
        Check if an emulator exists for specific iteration and feature.
        
        Args:
            iteration: Iteration number
            feature: Feature name
            
        Returns:
            True if emulator exists, False otherwise
        """
        return (iteration in self._emulators and 
                feature in self._emulators[iteration])
        
    def remove_emulator(self, iteration: int, feature: str) -> bool:
        """
        Remove a specific emulator.
        
        Args:
            iteration: Iteration number
            feature: Feature name
            
        Returns:
            True if emulator was removed, False if not found
        """
        if not self.has_emulator(iteration, feature):
            return False
            
        del self._emulators[iteration][feature]
        
        # Clean up empty iteration dict
        if not self._emulators[iteration]:
            del self._emulators[iteration]
            
        return True
        
    def remove_iteration(self, iteration: int):
        """
        Remove all emulators for a specific iteration.
        
        Args:
            iteration: Iteration number to remove
        """
        if iteration in self._emulators:
            del self._emulators[iteration]
            
    def clear(self):
        """Remove all emulators."""
        self._emulators.clear()
        
    def save_to_directory(self, directory_path: str):
        """
        Save all emulators to disk.
        
        Args:
            directory_path: Directory to save emulators in
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        for iteration in self.get_all_iterations():
            iteration_dir = os.path.join(directory_path, f"iteration_{iteration}")
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir)
                
            for feature, emulator in self.get_emulators_for_iteration(iteration).items():
                # Use pickle for serialization
                filename = f"{feature}_emulator.pkl"
                filepath = os.path.join(iteration_dir, filename)
                
                try:
                    with open(filepath, 'wb') as f:
                        pickle.dump(emulator, f)
                    logger.info(f"Saved emulator for iteration {iteration}, feature '{feature}' to {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save emulator for iteration {iteration}, feature '{feature}': {e}")
                    
    def load_from_directory(self, directory_path: str):
        """
        Load emulators from disk.
        
        Args:
            directory_path: Directory containing saved emulators
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
            
        # Clear existing emulators
        self.clear()
        
        # Look for iteration directories
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path) and item.startswith("iteration_"):
                try:
                    iteration = int(item.split("_")[1])
                except (IndexError, ValueError):
                    logger.warning(f"Skipping directory with invalid name: {item}")
                    continue
                    
                # Load emulators from iteration directory
                for filename in os.listdir(item_path):
                    if filename.endswith("_emulator.pkl"):
                        feature = filename.replace("_emulator.pkl", "")
                        filepath = os.path.join(item_path, filename)
                        
                        try:
                            with open(filepath, 'rb') as f:
                                emulator = pickle.load(f)
                            self.add_emulator(iteration, feature, emulator)
                            logger.info(f"Loaded emulator for iteration {iteration}, feature '{feature}' from {filepath}")
                        except Exception as e:
                            logger.error(f"Failed to load emulator from {filepath}: {e}")
                            
    def copy(self) -> 'EmulatorBank':
        """
        Create a deep copy of the emulator bank.
        
        Returns:
            New EmulatorBank instance with copied emulators
        """
        new_bank = EmulatorBank()
        new_bank._emulators = copy.deepcopy(self._emulators)
        return new_bank
        
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics about the emulator bank.
        
        Returns:
            Dict with summary information
        """
        total_emulators = sum(len(emulators) for emulators in self._emulators.values())
        
        return {
            'total_iterations': len(self._emulators),
            'total_emulators': total_emulators,
            'iterations': self.get_all_iterations(),
            'all_features': self.get_all_features(),
            'emulators_per_iteration': {
                iteration: len(emulators) 
                for iteration, emulators in self._emulators.items()
            }
        }
        
    def __len__(self) -> int:
        """Return total number of emulators across all iterations."""
        return sum(len(emulators) for emulators in self._emulators.values())
        
    def __contains__(self, key) -> bool:
        """
        Check if iteration or (iteration, feature) exists.
        
        Args:
            key: Either iteration number or (iteration, feature) tuple
            
        Returns:
            True if key exists, False otherwise
        """
        if isinstance(key, int):
            return key in self._emulators
        elif isinstance(key, tuple) and len(key) == 2:
            iteration, feature = key
            return self.has_emulator(iteration, feature)
        else:
            return False
            
    def __repr__(self) -> str:
        """String representation."""
        iterations = self.get_all_iterations()
        total_emulators = len(self)
        return f"EmulatorBank({len(iterations)} iterations, {total_emulators} emulators total)"