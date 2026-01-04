#!/usr/bin/env python3
"""
Model Versioning System
=======================
Manages model versions with semantic versioning and manifest tracking.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re


@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    version: str  # e.g., "v1.0.0"
    file: str  # Model filename
    accuracy: Optional[float] = None
    trained_date: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    dataset: Optional[str] = None
    metrics: Optional[Dict] = None


@dataclass
class ModelInfo:
    """Represents all versions of a model."""
    model_key: str  # e.g., "dlstm_ADA-EUR_1H_20240101-20251231"
    latest: str  # Latest version string
    versions: Dict[str, ModelVersion]  # version -> ModelVersion


class ModelVersionManager:
    """Manages model versions and manifest.json."""
    
    def __init__(self, manifest_path: Optional[Path] = None, models_dir: Optional[Path] = None):
        """
        Initialize version manager.
        
        Args:
            manifest_path: Path to manifest.json (default: models/manifest.json)
            models_dir: Directory containing model files (default: models/)
        """
        if manifest_path is None:
            # Default to models/manifest.json relative to this file
            base_dir = Path(__file__).parent
            manifest_path = base_dir / 'models' / 'manifest.json'
        
        if models_dir is None:
            models_dir = manifest_path.parent
        
        self.manifest_path = Path(manifest_path)
        self.models_dir = Path(models_dir)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create manifest
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load manifest.json or return empty dict."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load manifest.json: {e}")
                return {}
        return {}
    
    def _save_manifest(self):
        """Save manifest.json to disk."""
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            print(f"Error saving manifest.json: {e}")
            raise
    
    def _get_model_key(self, model_name: str, dataset_name: str) -> str:
        """Generate model key from model name and dataset."""
        return f"{model_name}_{dataset_name}"
    
    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """
        Parse version string to (major, minor, patch).
        
        Args:
            version_str: Version string like "v1.0.0" or "1.0.0"
            
        Returns:
            Tuple of (major, minor, patch)
        """
        # Remove 'v' prefix if present
        version_str = version_str.lstrip('v')
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}. Use v{major}.{minor}.{patch}")
        
        return tuple(int(p) for p in parts)
    
    def _format_version(self, major: int, minor: int, patch: int) -> str:
        """Format version tuple to string."""
        return f"v{major}.{minor}.{patch}"
    
    def get_latest_version(self, model_name: str, dataset_name: str) -> Optional[str]:
        """
        Get latest version for a model.
        
        Args:
            model_name: Model type (e.g., 'dlstm')
            dataset_name: Dataset identifier
            
        Returns:
            Latest version string or None if no versions exist
        """
        model_key = self._get_model_key(model_name, dataset_name)
        
        if model_key in self.manifest:
            return self.manifest[model_key].get('latest')
        
        return None
    
    def get_next_version(
        self,
        model_name: str,
        dataset_name: str,
        version_type: str = 'minor'
    ) -> str:
        """
        Get next version number for a model.
        
        Args:
            model_name: Model type
            dataset_name: Dataset identifier
            version_type: 'major', 'minor', or 'patch'
            
        Returns:
            Next version string (e.g., 'v1.1.0')
        """
        latest = self.get_latest_version(model_name, dataset_name)
        
        if latest is None:
            # First version
            return "v1.0.0"
        
        # Parse current version
        major, minor, patch = self._parse_version(latest)
        
        # Increment based on type
        if version_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif version_type == 'minor':
            minor += 1
            patch = 0
        elif version_type == 'patch':
            patch += 1
        else:
            raise ValueError(f"Invalid version_type: {version_type}. Use 'major', 'minor', or 'patch'")
        
        return self._format_version(major, minor, patch)
    
    def register_version(
        self,
        model_name: str,
        dataset_name: str,
        version: str,
        model_file: str,
        accuracy: Optional[float] = None,
        hyperparameters: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ):
        """
        Register a new model version in manifest.
        
        Args:
            model_name: Model type
            dataset_name: Dataset identifier
            version: Version string (e.g., 'v1.0.0')
            model_file: Model filename
            accuracy: Validation accuracy
            hyperparameters: Hyperparameters used
            metrics: Additional metrics
        """
        model_key = self._get_model_key(model_name, dataset_name)
        
        # Initialize model entry if doesn't exist
        if model_key not in self.manifest:
            self.manifest[model_key] = {
                'latest': version,
                'versions': {}
            }
        
        # Create version entry
        version_entry = {
            'file': model_file,
            'accuracy': accuracy,
            'trained_date': datetime.now().strftime('%Y-%m-%d'),
            'hyperparameters': hyperparameters or {},
            'dataset': dataset_name,
            'metrics': metrics or {}
        }
        
        # Add version
        self.manifest[model_key]['versions'][version] = version_entry
        
        # Update latest if this is newer
        current_latest = self.manifest[model_key].get('latest')
        if current_latest is None or self._is_version_newer(version, current_latest):
            self.manifest[model_key]['latest'] = version
        
        # Save manifest
        self._save_manifest()
    
    def _is_version_newer(self, version1: str, version2: str) -> bool:
        """Check if version1 is newer than version2."""
        v1 = self._parse_version(version1)
        v2 = self._parse_version(version2)
        return v1 > v2
    
    def get_version_info(
        self,
        model_name: str,
        dataset_name: str,
        version: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get information about a specific version.
        
        Args:
            model_name: Model type
            dataset_name: Dataset identifier
            version: Version string (None = latest)
            
        Returns:
            Version info dict or None if not found
        """
        model_key = self._get_model_key(model_name, dataset_name)
        
        if model_key not in self.manifest:
            return None
        
        if version is None:
            version = self.manifest[model_key].get('latest')
            if version is None:
                return None
        
        versions = self.manifest[model_key].get('versions', {})
        return versions.get(version)
    
    def list_versions(self, model_name: str, dataset_name: str) -> List[str]:
        """
        List all versions for a model.
        
        Args:
            model_name: Model type
            dataset_name: Dataset identifier
            
        Returns:
            List of version strings, sorted newest first
        """
        model_key = self._get_model_key(model_name, dataset_name)
        
        if model_key not in self.manifest:
            return []
        
        versions = list(self.manifest[model_key].get('versions', {}).keys())
        
        # Sort by version number (newest first)
        def version_key(v):
            major, minor, patch = self._parse_version(v)
            return (major, minor, patch)
        
        versions.sort(key=version_key, reverse=True)
        return versions
    
    def get_model_file(
        self,
        model_name: str,
        dataset_name: str,
        version: Optional[str] = None,
        task: str = 'classification'
    ) -> Optional[Path]:
        """
        Get path to model file for a specific version.
        
        Args:
            model_name: Model type
            dataset_name: Dataset identifier
            version: Version string (None = latest)
            task: Task type ('classification' or 'regression')
            
        Returns:
            Path to model file or None if not found
        """
        if version is None:
            version = self.get_latest_version(model_name, dataset_name)
            if version is None:
                # Fallback to old naming (backward compatibility)
                pattern = f"{model_name}_{dataset_name}_{task}.pth"
                model_path = self.models_dir / pattern
                if model_path.exists():
                    return model_path
                return None
        
        # Get version info
        version_info = self.get_version_info(model_name, dataset_name, version)
        if version_info is None:
            return None
        
        # Get file from manifest
        model_file = version_info.get('file')
        if model_file:
            model_path = self.models_dir / model_file
            if model_path.exists():
                return model_path
        
        # Fallback: construct filename
        pattern = f"{model_name}_{dataset_name}_{version}_{task}.pth"
        model_path = self.models_dir / pattern
        if model_path.exists():
            return model_path
        
        return None
    
    def initialize_from_existing_models(self):
        """
        Initialize manifest from existing model files (backward compatibility).
        Scans models directory and creates v1.0.0 entries for existing models.
        """
        print("Initializing model versioning from existing models...")
        
        # Pattern: {model}_{dataset}_{task}.pth
        pattern = re.compile(r'^(\w+)_([^_]+(?:_[^_]+)*)_(classification|regression)\.pth$')
        
        found_models = []
        for model_file in self.models_dir.glob('*.pth'):
            match = pattern.match(model_file.name)
            if match:
                model_name, dataset_name, task = match.groups()
                found_models.append((model_name, dataset_name, task, model_file.name))
        
        # Register as v1.0.0 if not already in manifest
        for model_name, dataset_name, task, filename in found_models:
            model_key = self._get_model_key(model_name, dataset_name)
            
            # Check if already registered
            if model_key in self.manifest:
                continue
            
            # Register as v1.0.0
            print(f"  Registering {model_name} {dataset_name} as v1.0.0")
            self.register_version(
                model_name=model_name,
                dataset_name=dataset_name,
                version="v1.0.0",
                model_file=filename
            )
        
        if found_models:
            print(f"✓ Initialized {len(found_models)} model(s) as v1.0.0")
        else:
            print("  No existing models found to initialize")


# Global instance (can be overridden)
_default_manager: Optional[ModelVersionManager] = None


def get_version_manager(manifest_path: Optional[Path] = None) -> ModelVersionManager:
    """Get or create default version manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelVersionManager(manifest_path)
    return _default_manager


if __name__ == '__main__':
    # Test/initialize versioning
    manager = ModelVersionManager()
    manager.initialize_from_existing_models()
    print(f"\nManifest saved to: {manager.manifest_path}")



