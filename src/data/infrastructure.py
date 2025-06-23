"""
Data Infrastructure Module

This module provides comprehensive data storage and management infrastructure
for the macro-regime model project.
"""

from pathlib import Path
import os
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
import shutil
import hashlib
import pickle

# Import the feature store and metadata tracker
from .feature_store import FeatureStore
from .metadata_tracker import MetadataTracker, DataType, OperationType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessedDataCache:
    """
    Intelligent caching system for processed data with automatic expiration,
    versioning, and dependency tracking.
    """
    
    def __init__(self, cache_path: Path, max_cache_size_mb: int = 1000):
        """
        Initialize the processed data cache.
        
        Args:
            cache_path: Path to cache directory
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_path = cache_path
        self.cache_index_file = cache_path / "cache_index.json"
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Create cache directory and subdirectories
        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(cache_path / "data", exist_ok=True)
        os.makedirs(cache_path / "metadata", exist_ok=True)
        
        # Initialize or load cache index
        self._initialize_cache_index()
        
        logger.info(f"Processed data cache initialized at: {cache_path}")
    
    def _initialize_cache_index(self):
        """Initialize cache index if it doesn't exist."""
        if not self.cache_index_file.exists():
            cache_index = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_cleanup": datetime.now(timezone.utc).isoformat(),
                "cache_entries": {},
                "statistics": {
                    "total_entries": 0,
                    "total_size_bytes": 0,
                    "cache_hits": 0,
                    "cache_misses": 0
                }
            }
            
            with open(self.cache_index_file, 'w') as f:
                json.dump(cache_index, f, indent=2)
            
            logger.info("Cache index initialized")
    
    def _generate_cache_key(self, data_identifier: str, parameters: Dict[str, Any] = None) -> str:
        """Generate a unique cache key based on data identifier and parameters."""
        key_components = [data_identifier]
        
        if parameters:
            # Sort parameters for consistent hashing
            sorted_params = json.dumps(parameters, sort_keys=True)
            key_components.append(sorted_params)
        
        combined_key = "|".join(key_components)
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    def store(self, 
              data: pd.DataFrame, 
              data_identifier: str,
              parameters: Dict[str, Any] = None,
              expiry_hours: int = 24,
              metadata: Dict[str, Any] = None) -> str:
        """
        Store processed data in cache with expiration and metadata.
        
        Args:
            data: DataFrame to cache
            data_identifier: Unique identifier for the data
            parameters: Processing parameters used to generate the data
            expiry_hours: Hours until cache entry expires
            metadata: Additional metadata to store
        
        Returns:
            str: Cache key for the stored data
        """
        try:
            cache_key = self._generate_cache_key(data_identifier, parameters)
            
            # Store data file
            data_file = self.cache_path / "data" / f"{cache_key}.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Store metadata
            metadata_file = self.cache_path / "metadata" / f"{cache_key}.json"
            entry_metadata = {
                "cache_key": cache_key,
                "data_identifier": data_identifier,
                "parameters": parameters or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).isoformat(),
                "data_shape": list(data.shape),
                "data_columns": list(data.columns),
                "file_size_bytes": os.path.getsize(data_file),
                "metadata": metadata or {},
                "access_count": 0,
                "last_accessed": datetime.now(timezone.utc).isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(entry_metadata, f, indent=2)
            
            # Update cache index
            self._update_cache_index(cache_key, entry_metadata)
            
            # Check cache size and cleanup if needed
            self._cleanup_if_needed()
            
            logger.info(f"Data cached successfully with key: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error storing data in cache: {e}")
            raise
    
    def retrieve(self, 
                 data_identifier: str, 
                 parameters: Dict[str, Any] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve processed data from cache if available and not expired.
        
        Args:
            data_identifier: Unique identifier for the data
            parameters: Processing parameters used to generate the data
        
        Returns:
            Optional[pd.DataFrame]: Cached data if found and valid, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(data_identifier, parameters)
            
            # Check if entry exists and is valid
            if not self._is_cache_valid(cache_key):
                self._update_cache_statistics("cache_misses")
                return None
            
            # Load data
            data_file = self.cache_path / "data" / f"{cache_key}.pkl"
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Update access statistics
            self._update_access_statistics(cache_key)
            self._update_cache_statistics("cache_hits")
            
            logger.info(f"Cache hit for key: {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving data from cache: {e}")
            self._update_cache_statistics("cache_misses")
            return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry exists and is not expired."""
        try:
            metadata_file = self.cache_path / "metadata" / f"{cache_key}.json"
            data_file = self.cache_path / "data" / f"{cache_key}.pkl"
            
            if not metadata_file.exists() or not data_file.exists():
                return False
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            
            return expires_at > datetime.now(timezone.utc)
            
        except Exception:
            return False
    
    def _update_cache_index(self, cache_key: str, entry_metadata: Dict[str, Any]):
        """Update the cache index with new entry information."""
        try:
            with open(self.cache_index_file, 'r') as f:
                index = json.load(f)
            
            index["cache_entries"][cache_key] = {
                "data_identifier": entry_metadata["data_identifier"],
                "created_at": entry_metadata["created_at"],
                "expires_at": entry_metadata["expires_at"],
                "file_size_bytes": entry_metadata["file_size_bytes"],
                "access_count": entry_metadata["access_count"]
            }
            
            index["statistics"]["total_entries"] = len(index["cache_entries"])
            index["statistics"]["total_size_bytes"] += entry_metadata["file_size_bytes"]
            
            with open(self.cache_index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating cache index: {e}")
    
    def _update_access_statistics(self, cache_key: str):
        """Update access statistics for a cache entry."""
        try:
            metadata_file = self.cache_path / "metadata" / f"{cache_key}.json"
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata["access_count"] += 1
            metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating access statistics: {e}")
    
    def _update_cache_statistics(self, stat_type: str):
        """Update global cache statistics."""
        try:
            with open(self.cache_index_file, 'r') as f:
                index = json.load(f)
            
            index["statistics"][stat_type] += 1
            
            with open(self.cache_index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating cache statistics: {e}")
    
    def _cleanup_if_needed(self):
        """Clean up expired entries and manage cache size."""
        try:
            current_time = datetime.now(timezone.utc)
            
            with open(self.cache_index_file, 'r') as f:
                index = json.load(f)
            
            # Remove expired entries
            expired_keys = []
            for cache_key, entry_info in index["cache_entries"].items():
                expires_at = datetime.fromisoformat(entry_info["expires_at"])
                if expires_at <= current_time:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                self._remove_cache_entry(key)
            
            # Check cache size and remove oldest entries if needed
            self._manage_cache_size()
            
            # Update last cleanup time
            index["last_cleanup"] = current_time.isoformat()
            
            with open(self.cache_index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries if size limit exceeded."""
        try:
            total_size = sum(
                os.path.getsize(f) 
                for f in (self.cache_path / "data").glob("*.pkl") 
                if f.is_file()
            )
            
            if total_size <= self.max_cache_size_bytes:
                return
            
            # Get entries sorted by last access time (oldest first)
            entries_by_access = []
            for cache_key in os.listdir(self.cache_path / "metadata"):
                if cache_key.endswith(".json"):
                    key = cache_key[:-5]  # Remove .json extension
                    metadata_file = self.cache_path / "metadata" / cache_key
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    entries_by_access.append((
                        key,
                        datetime.fromisoformat(metadata["last_accessed"]),
                        metadata["file_size_bytes"]
                    ))
            
            # Sort by last accessed (oldest first)
            entries_by_access.sort(key=lambda x: x[1])
            
            # Remove oldest entries until under size limit
            removed_count = 0
            for cache_key, _, size in entries_by_access:
                if total_size <= self.max_cache_size_bytes:
                    break
                
                self._remove_cache_entry(cache_key)
                total_size -= size
                removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} entries to manage cache size")
            
        except Exception as e:
            logger.error(f"Error managing cache size: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its associated files."""
        try:
            # Remove data file
            data_file = self.cache_path / "data" / f"{cache_key}.pkl"
            if data_file.exists():
                data_file.unlink()
            
            # Remove metadata file
            metadata_file = self.cache_path / "metadata" / f"{cache_key}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Update index
            with open(self.cache_index_file, 'r') as f:
                index = json.load(f)
            
            if cache_key in index["cache_entries"]:
                del index["cache_entries"][cache_key]
                index["statistics"]["total_entries"] = len(index["cache_entries"])
            
            with open(self.cache_index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error removing cache entry {cache_key}: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            with open(self.cache_index_file, 'r') as f:
                index = json.load(f)
            
            stats = index["statistics"].copy()
            
            # Calculate additional metrics
            if stats["cache_hits"] + stats["cache_misses"] > 0:
                hit_rate = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
                stats["hit_rate_percentage"] = round(hit_rate * 100, 2)
            else:
                stats["hit_rate_percentage"] = 0.0
            
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
            stats["max_size_mb"] = round(self.max_cache_size_bytes / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}


class DataInfrastructure:
    """
    Comprehensive data infrastructure management system for macro-regime analysis.
    """
    
    def __init__(self, base_path: str = "./Data"):
        """Initialize the data infrastructure with proper directory structure."""
        self.base_path = Path(base_path)
        
        # Define main data directories
        self.raw_data_path = self.base_path / "raw"
        self.processed_data_path = self.base_path / "processed"
        self.feature_store_path = self.base_path / "features"
        self.metadata_path = self.base_path / "metadata"
        
        # Data type subdirectories for better organization
        self.data_types = ["asset", "macro", "regime", "portfolio"]
        
        # Create the complete directory structure
        self._create_directories()
        
        # Initialize metadata tracking
        self.metadata_file = self.metadata_path / "data_registry.json"
        self._initialize_metadata()
        
        # Initialize processed data cache
        cache_path = self.processed_data_path / "cache"
        self.cache = ProcessedDataCache(cache_path)
        
        # Initialize feature store
        self.feature_store = FeatureStore(self.feature_store_path)
        
        # Initialize metadata tracker
        self.metadata_tracker = MetadataTracker(self.metadata_path)
        
        logger.info(f"Data infrastructure initialized at: {self.base_path}")
    
    def _create_directories(self):
        """Create the necessary directory structure with proper organization."""
        try:
            # Create main directories
            main_dirs = [
                self.raw_data_path,
                self.processed_data_path, 
                self.feature_store_path,
                self.metadata_path
            ]
            
            for directory in main_dirs:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created/verified directory: {directory}")
            
            # Create subdirectories for different data types
            for data_type in self.data_types:
                # Raw data subdirectories
                raw_subdir = self.raw_data_path / data_type
                os.makedirs(raw_subdir, exist_ok=True)
                
                # Processed data subdirectories
                processed_subdir = self.processed_data_path / data_type
                os.makedirs(processed_subdir, exist_ok=True)
                
                # Feature store subdirectories (handled by FeatureStore)
                
                logger.info(f"Created subdirectories for data type: {data_type}")
            
            logger.info("Complete directory structure created successfully")
            
        except Exception as e:
            logger.error(f"Error creating directory structure: {e}")
            raise
    
    def _initialize_metadata(self):
        """Initialize metadata tracking system."""
        try:
            if not self.metadata_file.exists():
                initial_metadata = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0.0",
                    "data_registry": {},
                    "schemas": {},
                    "lineage": {}
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(initial_metadata, f, indent=2)
                
                logger.info("Metadata registry initialized")
            else:
                logger.info("Metadata registry already exists")
                
        except Exception as e:
            logger.error(f"Error initializing metadata: {e}")
            raise
    
    def get_raw_data_path(self, data_type: str = None) -> Path:
        """Get the path to raw data directory."""
        if data_type:
            return self.raw_data_path / data_type
        return self.raw_data_path
    
    def get_processed_data_path(self, data_type: str = None) -> Path:
        """Get the path to processed data directory."""
        if data_type:
            return self.processed_data_path / data_type
        return self.processed_data_path
    
    def get_feature_store_path(self, feature_type: str = None) -> Path:
        """Get the path to feature store directory."""
        if feature_type:
            return self.feature_store_path / feature_type
        return self.feature_store_path
    
    # Cache methods
    def cache_processed_data(self, 
                           data: pd.DataFrame,
                           data_identifier: str,
                           parameters: Dict[str, Any] = None,
                           expiry_hours: int = 24,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Cache processed data with intelligent expiration and retrieval.
        
        Args:
            data: DataFrame to cache
            data_identifier: Unique identifier for the data
            parameters: Processing parameters used
            expiry_hours: Hours until expiration
            metadata: Additional metadata
        
        Returns:
            str: Cache key
        """
        return self.cache.store(data, data_identifier, parameters, expiry_hours, metadata)
    
    def get_cached_data(self, 
                       data_identifier: str,
                       parameters: Dict[str, Any] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve cached processed data if available.
        
        Args:
            data_identifier: Unique identifier for the data
            parameters: Processing parameters used
        
        Returns:
            Optional[pd.DataFrame]: Cached data if found, None otherwise
        """
        return self.cache.retrieve(data_identifier, parameters)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self.cache.get_cache_statistics()
    
    # Feature store methods
    def store_feature(self,
                     feature_data: pd.DataFrame,
                     feature_name: str,
                     feature_type: str = "engineered",
                     version: str = "1.0.0",
                     description: str = "",
                     metadata: Dict[str, Any] = None,
                     tags: List[str] = None) -> str:
        """
        Store a feature in the feature store.
        
        Args:
            feature_data: DataFrame containing the feature data
            feature_name: Unique name for the feature
            feature_type: Type of feature
            version: Version string
            description: Description
            metadata: Additional metadata
            tags: Tags for categorization
        
        Returns:
            str: Feature ID
        """
        return self.feature_store.store_feature(
            feature_data, feature_name, feature_type, version, description, metadata, tags
        )
    
    def retrieve_feature(self,
                        feature_name: str,
                        version: str = None) -> Optional[pd.DataFrame]:
        """
        Retrieve a feature from the feature store.
        
        Args:
            feature_name: Name of the feature
            version: Specific version (latest if None)
        
        Returns:
            Optional[pd.DataFrame]: Feature data if found
        """
        return self.feature_store.retrieve_feature(feature_name, version)
    
    def list_features(self,
                     feature_type: str = None,
                     tags: List[str] = None) -> List[Dict[str, Any]]:
        """List available features with optional filtering."""
        return self.feature_store.list_features(feature_type, tags)
    
    def create_feature_group(self,
                           group_name: str,
                           feature_names: List[str],
                           description: str = "",
                           metadata: Dict[str, Any] = None) -> str:
        """Create a logical grouping of related features."""
        return self.feature_store.create_feature_group(
            group_name, feature_names, description, metadata
        )
    
    def retrieve_feature_group(self, group_name: str) -> Optional[pd.DataFrame]:
        """Retrieve all features in a group as a combined DataFrame."""
        return self.feature_store.retrieve_feature_group(group_name)
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        return self.feature_store.get_feature_statistics()
    
    # Metadata tracking methods
    def register_data_node(self, name: str, data_type: DataType, location: str, 
                          metadata: Dict[str, Any] = None, data: pd.DataFrame = None) -> str:
        """Register a new data node in the metadata tracking system."""
        return self.metadata_tracker.register_data_node(name, data_type, location, metadata, data)
    
    def record_transformation(self, operation_type: OperationType, input_nodes: List[str],
                            output_nodes: List[str], parameters: Dict[str, Any] = None,
                            duration_seconds: float = 0.0, success: bool = True) -> str:
        """Record a data transformation in the metadata tracking system."""
        return self.metadata_tracker.record_transformation(
            operation_type, input_nodes, output_nodes, parameters, duration_seconds, success
        )
    
    def get_node_lineage(self, node_id: str) -> Dict[str, Any]:
        """Get lineage information for a specific data node."""
        return self.metadata_tracker.get_node_lineage(node_id)
    
    def search_data_nodes(self, name_pattern: str = None, data_type: DataType = None) -> List[Dict[str, Any]]:
        """Search for data nodes based on patterns and types."""
        return self.metadata_tracker.search_nodes(name_pattern, data_type)
    
    def get_metadata_statistics(self) -> Dict[str, Any]:
        """Get comprehensive metadata tracking statistics."""
        return self.metadata_tracker.get_statistics()
    
    def validate_directory_structure(self) -> Dict[str, bool]:
        """Validate that all required directories exist and are accessible."""
        validation_results = {}
        
        directories_to_check = [
            ("base", self.base_path),
            ("raw", self.raw_data_path),
            ("processed", self.processed_data_path),
            ("features", self.feature_store_path),
            ("metadata", self.metadata_path),
            ("cache", self.processed_data_path / "cache"),
            ("feature_engineered", self.feature_store_path / "engineered"),
            ("feature_metadata", self.feature_store_path / "metadata")
        ]
        
        for name, path in directories_to_check:
            try:
                validation_results[name] = path.exists() and path.is_dir()
            except Exception as e:
                validation_results[name] = False
                logger.warning(f"Directory {name} at {path} failed validation: {e}")
        
        return validation_results


def get_data_infrastructure(base_path: str = "./Data") -> DataInfrastructure:
    """Get a configured DataInfrastructure instance."""
    return DataInfrastructure(base_path=base_path) 