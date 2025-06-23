"""
Feature Store Module

This module provides comprehensive feature storage and management capabilities
for engineered features in the macro-regime model project.
"""

from pathlib import Path
import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import hashlib
import pickle
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Comprehensive feature store for managing engineered features with versioning,
    metadata tracking, and efficient storage/retrieval capabilities.
    """
    
    def __init__(self, base_path: Path):
        """
        Initialize the feature store.
        
        Args:
            base_path: Base path for feature store storage
        """
        self.base_path = base_path
        self.features_path = base_path / "engineered"
        self.metadata_path = base_path / "metadata"
        self.schemas_path = base_path / "schemas"
        self.index_file = self.metadata_path / "feature_index.json"
        
        # Create directory structure
        self._create_directories()
        
        # Initialize feature index
        self._initialize_feature_index()
        
        logger.info(f"Feature store initialized at: {base_path}")
    
    def _create_directories(self):
        """Create the necessary directory structure for feature store."""
        directories = [
            self.features_path,
            self.metadata_path, 
            self.schemas_path,
            self.features_path / "asset",
            self.features_path / "macro",
            self.features_path / "regime",
            self.features_path / "portfolio",
            self.features_path / "derived",
            self.features_path / "transformed"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Feature store directory structure created")
    
    def _initialize_feature_index(self):
        """Initialize the feature index if it doesn't exist."""
        if not self.index_file.exists():
            feature_index = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "features": {},
                "feature_groups": {},
                "statistics": {
                    "total_features": 0,
                    "total_versions": 0,
                    "storage_size_bytes": 0
                }
            }
            
            with open(self.index_file, 'w') as f:
                json.dump(feature_index, f, indent=2)
            
            logger.info("Feature index initialized")
    
    def store_feature(self,
                     feature_data: pd.DataFrame,
                     feature_name: str,
                     feature_type: str = "engineered",
                     version: str = "1.0.0",
                     description: str = "",
                     metadata: Dict[str, Any] = None,
                     tags: List[str] = None) -> str:
        """
        Store a feature with versioning and metadata.
        
        Args:
            feature_data: DataFrame containing the feature data
            feature_name: Unique name for the feature
            feature_type: Type of feature (engineered, derived, transformed, etc.)
            version: Version string for the feature
            description: Human-readable description
            metadata: Additional metadata
            tags: List of tags for categorization
        
        Returns:
            str: Feature ID for the stored feature
        """
        try:
            # Generate feature ID
            feature_id = self._generate_feature_id(feature_name, version)
            
            # Determine storage path based on feature type
            if feature_type in ["asset", "macro", "regime", "portfolio"]:
                storage_path = self.features_path / feature_type
            else:
                storage_path = self.features_path / feature_type
            
            os.makedirs(storage_path, exist_ok=True)
            
            # Store feature data
            feature_file = storage_path / f"{feature_id}.pkl"
            with open(feature_file, 'wb') as f:
                pickle.dump(feature_data, f)
            
            # Create feature metadata
            feature_metadata = {
                "feature_id": feature_id,
                "feature_name": feature_name,
                "feature_type": feature_type,
                "version": version,
                "description": description,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_shape": [int(x) for x in feature_data.shape],
                "data_columns": list(feature_data.columns),
                "data_types": {col: str(dtype) for col, dtype in feature_data.dtypes.items()},
                "file_path": str(feature_file),
                "file_size_bytes": int(os.path.getsize(feature_file)),
                "metadata": metadata or {},
                "tags": tags or [],
                "schema_hash": self._compute_schema_hash(feature_data),
                "statistics": self._compute_feature_statistics(feature_data)
            }
            
            # Store metadata
            metadata_file = self.metadata_path / f"{feature_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(feature_metadata, f, indent=2)
            
            # Store schema if new
            self._store_schema(feature_id, feature_data)
            
            # Update feature index
            self._update_feature_index(feature_id, feature_metadata)
            
            logger.info(f"Feature stored successfully: {feature_id}")
            return feature_id
            
        except Exception as e:
            logger.error(f"Error storing feature: {e}")
            raise
    
    def retrieve_feature(self,
                        feature_name: str,
                        version: str = None) -> Optional[pd.DataFrame]:
        """
        Retrieve a feature by name and optionally version.
        
        Args:
            feature_name: Name of the feature to retrieve
            version: Specific version (if None, returns latest)
        
        Returns:
            Optional[pd.DataFrame]: Feature data if found, None otherwise
        """
        try:
            # Find feature ID
            feature_id = self._find_feature_id(feature_name, version)
            if not feature_id:
                logger.warning(f"Feature not found: {feature_name} (version: {version})")
                return None
            
            # Load metadata to get file path
            metadata_file = self.metadata_path / f"{feature_id}.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load feature data
            feature_file = Path(metadata["file_path"])
            if not feature_file.exists():
                logger.error(f"Feature file not found: {feature_file}")
                return None
            
            with open(feature_file, 'rb') as f:
                feature_data = pickle.load(f)
            
            logger.info(f"Feature retrieved successfully: {feature_id}")
            return feature_data
            
        except Exception as e:
            logger.error(f"Error retrieving feature: {e}")
            return None
    
    def list_features(self,
                     feature_type: str = None,
                     tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        List available features with optional filtering.
        
        Args:
            feature_type: Filter by feature type
            tags: Filter by tags (features must have all specified tags)
        
        Returns:
            List[Dict]: List of feature information
        """
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            features = []
            for feature_id, feature_info in index["features"].items():
                # Apply filters
                if feature_type and feature_info["feature_type"] != feature_type:
                    continue
                
                if tags:
                    feature_tags = set(feature_info.get("tags", []))
                    if not set(tags).issubset(feature_tags):
                        continue
                
                features.append(feature_info)
            
            # Sort by creation date (newest first)
            features.sort(key=lambda x: x["created_at"], reverse=True)
            
            return features
            
        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return []
    
    def get_feature_versions(self, feature_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a specific feature.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            List[Dict]: List of version information
        """
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            versions = []
            for feature_id, feature_info in index["features"].items():
                if feature_info["feature_name"] == feature_name:
                    versions.append({
                        "version": feature_info["version"],
                        "created_at": feature_info["created_at"],
                        "description": feature_info["description"],
                        "feature_id": feature_id,
                        "data_shape": feature_info["data_shape"]
                    })
            
            # Sort by version (newest first)
            versions.sort(key=lambda x: x["created_at"], reverse=True)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting feature versions: {e}")
            return []
    
    def create_feature_group(self,
                           group_name: str,
                           feature_names: List[str],
                           description: str = "",
                           metadata: Dict[str, Any] = None) -> str:
        """
        Create a feature group that logically groups related features.
        
        Args:
            group_name: Name for the feature group
            feature_names: List of feature names to include
            description: Description of the group
            metadata: Additional metadata
        
        Returns:
            str: Group ID
        """
        try:
            group_id = f"group_{hashlib.md5(group_name.encode()).hexdigest()[:8]}"
            
            # Validate that all features exist
            valid_features = []
            for feature_name in feature_names:
                if self._find_feature_id(feature_name):
                    valid_features.append(feature_name)
                else:
                    logger.warning(f"Feature not found in group: {feature_name}")
            
            group_info = {
                "group_id": group_id,
                "group_name": group_name,
                "description": description,
                "features": valid_features,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            
            # Update index
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            index["feature_groups"][group_id] = group_info
            
            with open(self.index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Feature group created: {group_id}")
            return group_id
            
        except Exception as e:
            logger.error(f"Error creating feature group: {e}")
            raise
    
    def retrieve_feature_group(self, group_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve all features in a group as a joined DataFrame.
        
        Args:
            group_name: Name of the feature group
        
        Returns:
            Optional[pd.DataFrame]: Combined features or None if not found
        """
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # Find group
            group_info = None
            for group_id, info in index["feature_groups"].items():
                if info["group_name"] == group_name:
                    group_info = info
                    break
            
            if not group_info:
                logger.warning(f"Feature group not found: {group_name}")
                return None
            
            # Retrieve all features in the group
            features = []
            for feature_name in group_info["features"]:
                feature_data = self.retrieve_feature(feature_name)
                if feature_data is not None:
                    features.append(feature_data)
            
            if not features:
                logger.warning(f"No valid features found in group: {group_name}")
                return None
            
            # Combine features (assuming they have a common index)
            combined = features[0]
            for feature in features[1:]:
                combined = combined.join(feature, how='outer', rsuffix='_dup')
            
            logger.info(f"Feature group retrieved: {group_name}")
            return combined
            
        except Exception as e:
            logger.error(f"Error retrieving feature group: {e}")
            return None
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feature store statistics."""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            stats = index["statistics"].copy()
            
            # Calculate additional statistics
            feature_types = {}
            total_size = 0
            
            for feature_id, feature_info in index["features"].items():
                feature_type = feature_info["feature_type"]
                feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
                total_size += feature_info["file_size_bytes"]
            
            stats.update({
                "feature_types": feature_types,
                "total_groups": len(index["feature_groups"]),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "unique_feature_names": len(set(
                    info["feature_name"] for info in index["features"].values()
                ))
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return {}
    
    def _generate_feature_id(self, feature_name: str, version: str) -> str:
        """Generate a unique feature ID."""
        combined = f"{feature_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _find_feature_id(self, feature_name: str, version: str = None) -> Optional[str]:
        """Find feature ID by name and optionally version."""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            candidates = []
            for feature_id, feature_info in index["features"].items():
                if feature_info["feature_name"] == feature_name:
                    if version is None or feature_info["version"] == version:
                        candidates.append((feature_id, feature_info["created_at"]))
            
            if not candidates:
                return None
            
            # Return most recent if no version specified
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
            
        except Exception:
            return None
    
    def _compute_schema_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of DataFrame schema."""
        schema_info = {
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _store_schema(self, feature_id: str, data: pd.DataFrame):
        """Store feature schema information."""
        try:
            schema_hash = self._compute_schema_hash(data)
            schema_file = self.schemas_path / f"{schema_hash}.json"
            
            if not schema_file.exists():
                schema_info = {
                    "schema_hash": schema_hash,
                    "columns": list(data.columns),
                    "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                    "shape": [int(x) for x in data.shape],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "example_features": [feature_id]
                }
                
                with open(schema_file, 'w') as f:
                    json.dump(schema_info, f, indent=2)
            else:
                # Update existing schema with new feature reference
                with open(schema_file, 'r') as f:
                    schema_info = json.load(f)
                
                if feature_id not in schema_info["example_features"]:
                    schema_info["example_features"].append(feature_id)
                
                with open(schema_file, 'w') as f:
                    json.dump(schema_info, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error storing schema: {e}")
    
    def _compute_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics for feature data."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            null_counts = data.isnull().sum().to_dict()
            null_counts = {k: int(v) for k, v in null_counts.items()}
            
            stats = {
                "row_count": int(len(data)),
                "column_count": int(len(data.columns)),
                "null_counts": null_counts,
                "memory_usage_bytes": int(data.memory_usage(deep=True).sum())
            }
            
            # Add numeric column statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_summary = data[numeric_cols].describe().to_dict()
                # Convert numpy types to native Python types
                for col, col_stats in numeric_summary.items():
                    for stat, value in col_stats.items():
                        if pd.isna(value):
                            numeric_summary[col][stat] = None
                        else:
                            numeric_summary[col][stat] = float(value)
                stats["numeric_summary"] = numeric_summary
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing feature statistics: {e}")
            return {}
    
    def _update_feature_index(self, feature_id: str, feature_metadata: Dict[str, Any]):
        """Update the main feature index with new feature information."""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # Add feature to index
            index["features"][feature_id] = {
                "feature_name": feature_metadata["feature_name"],
                "feature_type": feature_metadata["feature_type"],
                "version": feature_metadata["version"],
                "description": feature_metadata["description"],
                "created_at": feature_metadata["created_at"],
                "data_shape": feature_metadata["data_shape"],
                "file_size_bytes": feature_metadata["file_size_bytes"],
                "tags": feature_metadata["tags"],
                "schema_hash": feature_metadata["schema_hash"]
            }
            
            # Update statistics
            index["statistics"]["total_features"] = len(index["features"])
            index["statistics"]["storage_size_bytes"] += feature_metadata["file_size_bytes"]
            
            # Count versions
            feature_names = set(info["feature_name"] for info in index["features"].values())
            total_versions = 0
            for name in feature_names:
                versions = self.get_feature_versions(name)
                total_versions += len(versions)
            index["statistics"]["total_versions"] = total_versions
            
            with open(self.index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating feature index: {e}")
            raise 