"""
Feature Store Integration System

This module provides a comprehensive feature store integration system for managing,
storing, versioning, and retrieving features in a machine learning pipeline.

Features:
- Feature metadata management with schemas
- Feature versioning and compatibility tracking
- Multiple storage backend support
- Query interfaces for feature retrieval
- Feature lineage and dependency tracking
- Configuration management
- Feature validation and quality checks

Author: Macro Regime Analysis Platform
Date: 2025-01-19
"""

import json
import logging
import pickle
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from uuid import uuid4

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"


class FeatureStatus(Enum):
    """Feature lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active" 
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class StorageBackend(Enum):
    """Supported storage backends."""
    FILE_SYSTEM = "file_system"
    SQLITE = "sqlite"
    MEMORY = "memory"


@dataclass
class FeatureMetadata:
    """Feature metadata schema."""
    feature_id: str
    name: str
    description: str
    feature_type: FeatureType
    version: str = "1.0.0"
    status: FeatureStatus = FeatureStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    source_tables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    transformation_logic: str = ""
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['feature_type'] = self.feature_type.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        
        # Convert numpy types to native Python types for JSON serialization
        result = self._convert_numpy_types(result)
        return result
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['feature_type'] = FeatureType(data['feature_type'])
        data['status'] = FeatureStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class FeatureQuery:
    """Feature query specification."""
    feature_names: List[str] = field(default_factory=list)
    feature_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    feature_types: List[FeatureType] = field(default_factory=list)
    status: Optional[FeatureStatus] = None
    version_pattern: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: Optional[int] = None
    include_deprecated: bool = False


@dataclass 
class FeatureStoreConfig:
    """Feature store configuration."""
    storage_backend: StorageBackend = StorageBackend.FILE_SYSTEM
    storage_path: str = ".feature_store"
    enable_versioning: bool = True
    enable_lineage_tracking: bool = True
    enable_quality_checks: bool = True
    max_versions_per_feature: int = 10
    auto_cleanup_days: int = 30
    compression: bool = True


class StorageInterface(ABC):
    """Abstract storage interface."""
    
    @abstractmethod
    def save_feature(self, feature_id: str, data: Any, metadata: FeatureMetadata) -> bool:
        """Save feature data and metadata."""
        pass
    
    @abstractmethod
    def load_feature(self, feature_id: str, version: Optional[str] = None) -> Tuple[Any, FeatureMetadata]:
        """Load feature data and metadata."""
        pass
    
    @abstractmethod
    def delete_feature(self, feature_id: str, version: Optional[str] = None) -> bool:
        """Delete feature data."""
        pass
    
    @abstractmethod
    def list_features(self, query: FeatureQuery) -> List[FeatureMetadata]:
        """List features matching query."""
        pass
    
    @abstractmethod
    def feature_exists(self, feature_id: str, version: Optional[str] = None) -> bool:
        """Check if feature exists."""
        pass


class FileSystemStorage(StorageInterface):
    """File system storage backend."""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.base_path = Path(config.storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "features").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "indexes").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_feature_path(self, feature_id: str, version: str) -> Path:
        """Get feature data file path."""
        return self.base_path / "features" / f"{feature_id}_v{version}.pkl"
    
    def _get_metadata_path(self, feature_id: str, version: str) -> Path:
        """Get metadata file path."""
        return self.base_path / "metadata" / f"{feature_id}_v{version}.json"
    
    def save_feature(self, feature_id: str, data: Any, metadata: FeatureMetadata) -> bool:
        """Save feature data and metadata to file system."""
        try:
            # Save feature data
            feature_path = self._get_feature_path(feature_id, metadata.version)
            with open(feature_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata_path = self._get_metadata_path(feature_id, metadata.version)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved feature {feature_id} v{metadata.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save feature {feature_id}: {e}")
            return False
    
    def load_feature(self, feature_id: str, version: Optional[str] = None) -> Tuple[Any, FeatureMetadata]:
        """Load feature data and metadata."""
        if version is None:
            version = self._get_latest_version(feature_id)
            if version is None:
                raise FileNotFoundError(f"Feature {feature_id} not found - no versions available")
        
        feature_path = self._get_feature_path(feature_id, version)
        metadata_path = self._get_metadata_path(feature_id, version)
        
        if not feature_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Feature {feature_id} v{version} not found")
        
        # Load data
        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata = FeatureMetadata.from_dict(metadata_dict)
        
        return data, metadata
    
    def delete_feature(self, feature_id: str, version: Optional[str] = None) -> bool:
        """Delete feature data and metadata."""
        try:
            if version is None:
                # Delete all versions
                versions = self._get_feature_versions(feature_id)
                for v in versions:
                    self._delete_version(feature_id, v)
            else:
                self._delete_version(feature_id, version)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete feature {feature_id}: {e}")
            return False
    
    def _delete_version(self, feature_id: str, version: str):
        """Delete specific version."""
        feature_path = self._get_feature_path(feature_id, version)
        metadata_path = self._get_metadata_path(feature_id, version)
        
        if feature_path.exists():
            feature_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    
    def list_features(self, query: FeatureQuery) -> List[FeatureMetadata]:
        """List features matching query criteria."""
        features = []
        metadata_dir = self.base_path / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = FeatureMetadata.from_dict(metadata_dict)
                
                if self._matches_query(metadata, query):
                    features.append(metadata)
                    
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        # Sort by creation date (newest first)
        features.sort(key=lambda x: x.created_at, reverse=True)
        
        if query.limit:
            features = features[:query.limit]
        
        return features
    
    def _matches_query(self, metadata: FeatureMetadata, query: FeatureQuery) -> bool:
        """Check if metadata matches query criteria."""
        # Feature names
        if query.feature_names and metadata.name not in query.feature_names:
            return False
        
        # Feature IDs
        if query.feature_ids and metadata.feature_id not in query.feature_ids:
            return False
        
        # Tags
        if query.tags and not any(tag in metadata.tags for tag in query.tags):
            return False
        
        # Feature types
        if query.feature_types and metadata.feature_type not in query.feature_types:
            return False
        
        # Status
        if query.status and metadata.status != query.status:
            return False
        
        # Deprecated features
        if not query.include_deprecated and metadata.status == FeatureStatus.DEPRECATED:
            return False
        
        # Date range
        if query.created_after and metadata.created_at < query.created_after:
            return False
        if query.created_before and metadata.created_at > query.created_before:
            return False
        
        return True
    
    def feature_exists(self, feature_id: str, version: Optional[str] = None) -> bool:
        """Check if feature exists."""
        if version is None:
            version = self._get_latest_version(feature_id)
        
        if version is None:
            return False
        
        feature_path = self._get_feature_path(feature_id, version)
        metadata_path = self._get_metadata_path(feature_id, version)
        
        return feature_path.exists() and metadata_path.exists()
    
    def _get_feature_versions(self, feature_id: str) -> List[str]:
        """Get all versions of a feature."""
        versions = []
        metadata_dir = self.base_path / "metadata"
        
        for metadata_file in metadata_dir.glob(f"{feature_id}_v*.json"):
            # Extract version from filename: feature_id_v1.0.0_uuid.json -> 1.0.0
            parts = metadata_file.stem.split('_v')
            if len(parts) >= 2:
                # Split by underscore and take first part (version before uuid)
                version_part = parts[1].split('_')[0]
                if '.' in version_part:  # Ensure it looks like a version
                    versions.append(version_part)
        
        return sorted(versions)
    
    def _get_latest_version(self, feature_id: str) -> Optional[str]:
        """Get the latest version of a feature."""
        versions = self._get_feature_versions(feature_id)
        return versions[-1] if versions else None


class MemoryStorage(StorageInterface):
    """In-memory storage backend for testing."""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.features: Dict[str, Any] = {}
        self.metadata: Dict[str, FeatureMetadata] = {}
        self.lock = threading.Lock()
    
    def _get_key(self, feature_id: str, version: str) -> str:
        """Generate storage key."""
        return f"{feature_id}_v{version}"
    
    def save_feature(self, feature_id: str, data: Any, metadata: FeatureMetadata) -> bool:
        """Save feature to memory."""
        key = self._get_key(feature_id, metadata.version)
        
        with self.lock:
            self.features[key] = data
            self.metadata[key] = metadata
        
        return True
    
    def load_feature(self, feature_id: str, version: Optional[str] = None) -> Tuple[Any, FeatureMetadata]:
        """Load feature from memory."""
        if version is None:
            version = self._get_latest_version(feature_id)
        
        key = self._get_key(feature_id, version)
        
        with self.lock:
            if key not in self.features:
                raise KeyError(f"Feature {feature_id} v{version} not found")
            
            return self.features[key], self.metadata[key]
    
    def delete_feature(self, feature_id: str, version: Optional[str] = None) -> bool:
        """Delete feature from memory."""
        with self.lock:
            if version is None:
                # Delete all versions
                keys_to_delete = [k for k in self.features.keys() if k.startswith(f"{feature_id}_v")]
                for key in keys_to_delete:
                    del self.features[key]
                    del self.metadata[key]
            else:
                key = self._get_key(feature_id, version)
                if key in self.features:
                    del self.features[key]
                    del self.metadata[key]
        
        return True
    
    def list_features(self, query: FeatureQuery) -> List[FeatureMetadata]:
        """List features in memory."""
        with self.lock:
            features = []
            for metadata in self.metadata.values():
                if self._matches_query(metadata, query):
                    features.append(metadata)
            
            features.sort(key=lambda x: x.created_at, reverse=True)
            
            if query.limit:
                features = features[:query.limit]
            
            return features
    
    def _matches_query(self, metadata: FeatureMetadata, query: FeatureQuery) -> bool:
        """Check if metadata matches query - same logic as FileSystemStorage."""
        if query.feature_names and metadata.name not in query.feature_names:
            return False
        if query.feature_ids and metadata.feature_id not in query.feature_ids:
            return False
        if query.tags and not any(tag in metadata.tags for tag in query.tags):
            return False
        if query.feature_types and metadata.feature_type not in query.feature_types:
            return False
        if query.status and metadata.status != query.status:
            return False
        if not query.include_deprecated and metadata.status == FeatureStatus.DEPRECATED:
            return False
        if query.created_after and metadata.created_at < query.created_after:
            return False
        if query.created_before and metadata.created_at > query.created_before:
            return False
        
        return True
    
    def feature_exists(self, feature_id: str, version: Optional[str] = None) -> bool:
        """Check if feature exists in memory."""
        if version is None:
            version = self._get_latest_version(feature_id)
        
        if version is None:
            return False
        
        key = self._get_key(feature_id, version)
        return key in self.features
    
    def _get_latest_version(self, feature_id: str) -> Optional[str]:
        """Get latest version from memory."""
        with self.lock:
            versions = []
            for key in self.features.keys():
                if key.startswith(f"{feature_id}_v"):
                    version = key.split('_v')[1]
                    versions.append(version)
            
            return sorted(versions)[-1] if versions else None


class FeatureStore:
    """Main feature store interface."""
    
    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or FeatureStoreConfig()
        self.storage = self._create_storage()
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature registry
        self.feature_registry: Dict[str, Set[str]] = {}  # feature_name -> set of versions
        self._refresh_registry()
    
    def _create_storage(self) -> StorageInterface:
        """Create storage backend based on configuration."""
        if self.config.storage_backend == StorageBackend.FILE_SYSTEM:
            return FileSystemStorage(self.config)
        elif self.config.storage_backend == StorageBackend.MEMORY:
            return MemoryStorage(self.config)
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
    
    def _refresh_registry(self):
        """Refresh feature registry."""
        try:
            query = FeatureQuery(include_deprecated=True)
            features = self.storage.list_features(query)
            
            self.feature_registry.clear()
            for feature in features:
                if feature.name not in self.feature_registry:
                    self.feature_registry[feature.name] = set()
                self.feature_registry[feature.name].add(feature.version)
                
        except Exception as e:
            self.logger.warning(f"Failed to refresh registry: {e}")
    
    def save_feature(
        self,
        name: str,
        data: Union[pd.DataFrame, pd.Series, np.ndarray, List, Dict],
        description: str = "",
        feature_type: Optional[FeatureType] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        transformation_logic: str = "",
        validation_rules: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Save a feature to the store.
        
        Args:
            name: Feature name
            data: Feature data
            description: Feature description
            feature_type: Type of feature
            version: Feature version (auto-generated if None)
            tags: Feature tags
            dependencies: List of dependent feature IDs
            transformation_logic: Description of how feature was created
            validation_rules: Validation rules for the feature
            **kwargs: Additional metadata
            
        Returns:
            feature_id: Unique feature identifier
        """
        # Auto-detect feature type if not provided
        if feature_type is None:
            feature_type = self._detect_feature_type(data)
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)
        
        # Generate unique feature ID
        feature_id = self._generate_feature_id(name, version)
        
        # Calculate feature statistics
        statistics = self._calculate_statistics(data)
        
        # Create metadata
        metadata = FeatureMetadata(
            feature_id=feature_id,
            name=name,
            description=description,
            feature_type=feature_type,
            version=version,
            tags=tags or [],
            dependencies=dependencies or [],
            transformation_logic=transformation_logic,
            validation_rules=validation_rules or {},
            statistics=statistics,
            **kwargs
        )
        
        # Validate feature if rules are provided
        if validation_rules:
            self._validate_feature(data, validation_rules)
        
        # Save to storage
        success = self.storage.save_feature(feature_id, data, metadata)
        
        if success:
            # Update registry
            if name not in self.feature_registry:
                self.feature_registry[name] = set()
            self.feature_registry[name].add(version)
            
            self.logger.info(f"Saved feature '{name}' v{version} with ID {feature_id}")
            return feature_id
        else:
            raise RuntimeError(f"Failed to save feature '{name}'")
    
    def load_feature(
        self,
        name: Optional[str] = None,
        feature_id: Optional[str] = None,
        version: Optional[str] = None
    ) -> Tuple[Any, FeatureMetadata]:
        """
        Load a feature from the store.
        
        Args:
            name: Feature name (if feature_id not provided)
            feature_id: Feature ID
            version: Specific version (latest if None)
            
        Returns:
            (data, metadata): Feature data and metadata
        """
        if feature_id is None:
            if name is None:
                raise ValueError("Must provide either feature_id or name")
            
            # Find feature ID by name and version
            feature_id = self._find_feature_id(name, version)
            if feature_id is None:
                raise KeyError(f"Feature '{name}' v{version or 'latest'} not found")
        
        # For storage.load_feature, don't pass version when loading by feature_id 
        # since feature_id already contains version info
        if version is None:
            return self.storage.load_feature(feature_id)
        else:
            return self.storage.load_feature(feature_id, version)
    
    def delete_feature(
        self,
        name: Optional[str] = None,
        feature_id: Optional[str] = None,
        version: Optional[str] = None
    ) -> bool:
        """Delete a feature from the store."""
        if feature_id is None:
            if name is None:
                raise ValueError("Must provide either feature_id or name")
            
            feature_id = self._find_feature_id(name, version)
            if feature_id is None:
                raise KeyError(f"Feature '{name}' not found")
        
        success = self.storage.delete_feature(feature_id, version)
        
        if success:
            self._refresh_registry()
        
        return success
    
    def query_features(self, query: FeatureQuery) -> List[FeatureMetadata]:
        """Query features based on criteria."""
        return self.storage.list_features(query)
    
    def list_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self.feature_registry.keys())
    
    def get_feature_versions(self, name: str) -> List[str]:
        """Get all versions of a feature."""
        return sorted(list(self.feature_registry.get(name, set())))
    
    def feature_exists(self, name: str, version: Optional[str] = None) -> bool:
        """Check if a feature exists."""
        if name not in self.feature_registry:
            return False
        
        if version is None:
            return len(self.feature_registry[name]) > 0
        
        return version in self.feature_registry[name]
    
    def get_feature_lineage(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get feature lineage information."""
        try:
            _, metadata = self.load_feature(name=name, version=version)
            
            # Build dependency chain with error handling
            dependency_chain = []
            if metadata.dependencies:
                dependency_chain = self._build_dependency_chain(metadata.dependencies)
            
            lineage = {
                'feature_id': metadata.feature_id,
                'name': metadata.name,
                'version': metadata.version,
                'dependencies': metadata.dependencies,
                'transformation_logic': metadata.transformation_logic,
                'created_at': metadata.created_at,
                'dependency_chain': dependency_chain
            }
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Failed to get lineage for {name}: {e}")
            return {
                'feature_id': 'unknown',
                'name': name,
                'version': version or 'unknown',
                'dependencies': [],
                'transformation_logic': '',
                'created_at': None,
                'dependency_chain': [],
                'error': str(e)
            }
    
    def _build_dependency_chain(self, dependencies: List[str], visited: Optional[Set[str]] = None) -> List[Dict]:
        """Build recursive dependency chain."""
        if visited is None:
            visited = set()
        
        chain = []
        
        for dep_id in dependencies:
            if dep_id in visited:
                continue  # Avoid circular dependencies
            
            visited.add(dep_id)
            
            try:
                # Use the main load_feature method instead of storage directly
                _, dep_metadata = self.load_feature(feature_id=dep_id)
                dep_info = {
                    'feature_id': dep_metadata.feature_id,
                    'name': dep_metadata.name,
                    'version': dep_metadata.version,
                    'dependencies': self._build_dependency_chain(dep_metadata.dependencies, visited.copy())
                }
                chain.append(dep_info)
                
            except Exception as e:
                self.logger.warning(f"Could not load dependency {dep_id}: {e}")
                # Include partial info for failed dependencies
                chain.append({
                    'feature_id': dep_id,
                    'name': 'unknown',
                    'version': 'unknown',
                    'error': str(e),
                    'dependencies': []
                })
        
        return chain
    
    def _detect_feature_type(self, data: Any) -> FeatureType:
        """Auto-detect feature type from data."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if len(data) == 0:
                return FeatureType.OBJECT
            
            # Check first non-null value
            sample = data.iloc[0] if isinstance(data, pd.Series) else data.iloc[0, 0]
            
            if pd.api.types.is_numeric_dtype(type(sample)):
                return FeatureType.NUMERICAL
            elif pd.api.types.is_bool_dtype(type(sample)):
                return FeatureType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(type(sample)):
                return FeatureType.DATETIME
            else:
                return FeatureType.CATEGORICAL
                
        elif isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.number):
                return FeatureType.NUMERICAL
            elif np.issubdtype(data.dtype, np.bool_):
                return FeatureType.BOOLEAN
            else:
                return FeatureType.ARRAY
                
        elif isinstance(data, (list, tuple)):
            return FeatureType.ARRAY
        else:
            return FeatureType.OBJECT
    
    def _generate_version(self, name: str) -> str:
        """Generate next version number for a feature."""
        if name not in self.feature_registry:
            return "1.0.0"
        
        versions = self.feature_registry[name]
        if not versions:
            return "1.0.0"
        
        # Find highest version
        max_version = max(versions)
        
        # Parse and increment
        try:
            parts = max_version.split('.')
            major, minor, patch = map(int, parts)
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.0"
    
    def _generate_feature_id(self, name: str, version: str) -> str:
        """Generate unique feature ID."""
        return f"{name}_{version}_{str(uuid4())[:8]}"
    
    def _find_feature_id(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """Find feature ID by name and version."""
        if version is None:
            version = max(self.feature_registry.get(name, set()), default=None)
            if version is None:
                return None
        
        # Search for feature with matching name and version
        query = FeatureQuery(feature_names=[name])
        features = self.storage.list_features(query)
        
        for feature in features:
            if feature.name == name and feature.version == version:
                return feature.feature_id
        
        return None
    
    def _calculate_statistics(self, data: Any) -> Dict[str, Any]:
        """Calculate basic statistics for feature data."""
        stats = {}
        
        try:
            if isinstance(data, pd.DataFrame):
                stats['shape'] = data.shape
                stats['columns'] = list(data.columns)
                stats['null_count'] = data.isnull().sum().to_dict()
                
                # Numeric columns stats
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats['numeric_stats'] = data[numeric_cols].describe().to_dict()
                    
            elif isinstance(data, pd.Series):
                stats['shape'] = (len(data),)
                stats['null_count'] = data.isnull().sum()
                
                if pd.api.types.is_numeric_dtype(data):
                    stats['numeric_stats'] = data.describe().to_dict()
                    
            elif isinstance(data, np.ndarray):
                stats['shape'] = data.shape
                stats['dtype'] = str(data.dtype)
                
                if np.issubdtype(data.dtype, np.number):
                    stats['mean'] = float(np.mean(data))
                    stats['std'] = float(np.std(data))
                    stats['min'] = float(np.min(data))
                    stats['max'] = float(np.max(data))
                    
        except Exception as e:
            self.logger.warning(f"Could not calculate statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _validate_feature(self, data: Any, rules: Dict[str, Any]):
        """Validate feature data against rules."""
        try:
            # Example validation rules
            if 'min_value' in rules and isinstance(data, (pd.Series, np.ndarray)):
                if hasattr(data, 'min'):
                    if data.min() < rules['min_value']:
                        raise ValueError(f"Feature minimum {data.min()} below threshold {rules['min_value']}")
            
            if 'max_value' in rules and isinstance(data, (pd.Series, np.ndarray)):
                if hasattr(data, 'max'):
                    if data.max() > rules['max_value']:
                        raise ValueError(f"Feature maximum {data.max()} above threshold {rules['max_value']}")
            
            if 'required_columns' in rules and isinstance(data, pd.DataFrame):
                missing_cols = set(rules['required_columns']) - set(data.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                    
        except Exception as e:
            self.logger.error(f"Feature validation failed: {e}")
            raise


# Convenience functions
def create_feature_store(config: Optional[FeatureStoreConfig] = None) -> FeatureStore:
    """Create a new feature store instance."""
    return FeatureStore(config)


def create_file_feature_store(storage_path: str = ".feature_store") -> FeatureStore:
    """Create a file-based feature store."""
    config = FeatureStoreConfig(
        storage_backend=StorageBackend.FILE_SYSTEM,
        storage_path=storage_path
    )
    return FeatureStore(config)


def create_memory_feature_store() -> FeatureStore:
    """Create an in-memory feature store for testing."""
    config = FeatureStoreConfig(storage_backend=StorageBackend.MEMORY)
    return FeatureStore(config)


# Example usage
if __name__ == "__main__":
    # Create feature store
    store = create_file_feature_store()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Save feature
    feature_id = store.save_feature(
        name="sample_features",
        data=sample_data,
        description="Sample multi-column feature set",
        tags=["sample", "test"],
        transformation_logic="Generated random data for testing"
    )
    
    print(f"Saved feature with ID: {feature_id}")
    
    # Load feature
    loaded_data, metadata = store.load_feature(name="sample_features")
    print(f"Loaded feature: {metadata.name} v{metadata.version}")
    print(f"Data shape: {loaded_data.shape}")
    
    # Query features
    query = FeatureQuery(tags=["sample"])
    results = store.query_features(query)
    print(f"Found {len(results)} features with 'sample' tag")
    
    # Get lineage
    lineage = store.get_feature_lineage("sample_features")
    print(f"Feature lineage: {lineage}")
