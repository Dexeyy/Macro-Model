"""
Metadata Tracking Module for Data Lineage and Transformations
"""

from pathlib import Path
import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
import logging
import hashlib
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperationType(Enum):
    CREATE = "create"
    UPDATE = "update"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    MERGE = "merge"
    CACHE = "cache"
    FEATURE_ENGINEERING = "feature_engineering"


class DataType(Enum):
    RAW_DATA = "raw_data"
    PROCESSED_DATA = "processed_data"
    FEATURE = "feature"
    CACHE_ENTRY = "cache_entry"


@dataclass
class DataLineageNode:
    node_id: str
    name: str
    data_type: DataType
    location: str
    created_at: str
    metadata: Dict[str, Any]
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass
class DataTransformation:
    transformation_id: str
    operation_type: OperationType
    input_nodes: List[str]
    output_nodes: List[str]
    execution_time: str
    duration_seconds: float
    parameters: Dict[str, Any]
    success: bool = True


class MetadataTracker:
    """Comprehensive metadata tracking system."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.lineage_path = base_path / "lineage"
        self.transformations_path = base_path / "transformations"
        
        self.lineage_index_file = self.lineage_path / "lineage_index.json"
        self.transformation_index_file = self.transformations_path / "transformation_index.json"
        
        self._create_directories()
        self._initialize_indices()
        
        logger.info(f"Metadata tracker initialized at: {base_path}")
    
    def _create_directories(self):
        """Create directory structure."""
        directories = [
            self.lineage_path,
            self.transformations_path,
            self.lineage_path / "nodes",
            self.transformations_path / "operations"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_indices(self):
        """Initialize index files."""
        if not self.lineage_index_file.exists():
            lineage_index = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "nodes": {},
                "statistics": {"total_nodes": 0, "nodes_by_type": {}}
            }
            with open(self.lineage_index_file, 'w') as f:
                json.dump(lineage_index, f, indent=2)
        
        if not self.transformation_index_file.exists():
            transformation_index = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "transformations": {},
                "statistics": {"total_transformations": 0, "operations_by_type": {}}
            }
            with open(self.transformation_index_file, 'w') as f:
                json.dump(transformation_index, f, indent=2)
    
    def register_data_node(self, name: str, data_type: DataType, location: str, 
                          metadata: Dict[str, Any] = None, data: pd.DataFrame = None) -> str:
        """Register a new data node."""
        try:
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            
            checksum = None
            size_bytes = None
            if data is not None:
                checksum = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()
                size_bytes = int(data.memory_usage(deep=True).sum())
            
            lineage_node = DataLineageNode(
                node_id=node_id,
                name=name,
                data_type=data_type,
                location=location,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
                checksum=checksum,
                size_bytes=size_bytes
            )
            
            # Store node details
            node_file = self.lineage_path / "nodes" / f"{node_id}.json"
            with open(node_file, 'w') as f:
                json.dump(asdict(lineage_node), f, indent=2, default=str)
            
            # Update index
            self._update_lineage_index(node_id, lineage_node)
            
            logger.info(f"Data node registered: {node_id} ({name})")
            return node_id
            
        except Exception as e:
            logger.error(f"Error registering data node: {e}")
            raise
    
    def record_transformation(self, operation_type: OperationType, input_nodes: List[str],
                            output_nodes: List[str], parameters: Dict[str, Any] = None,
                            duration_seconds: float = 0.0, success: bool = True) -> str:
        """Record a data transformation."""
        try:
            transformation_id = f"transform_{uuid.uuid4().hex[:8]}"
            
            transformation = DataTransformation(
                transformation_id=transformation_id,
                operation_type=operation_type,
                input_nodes=input_nodes,
                output_nodes=output_nodes,
                execution_time=datetime.now(timezone.utc).isoformat(),
                duration_seconds=duration_seconds,
                parameters=parameters or {},
                success=success
            )
            
            # Store transformation
            transform_file = self.transformations_path / "operations" / f"{transformation_id}.json"
            with open(transform_file, 'w') as f:
                json.dump(asdict(transformation), f, indent=2, default=str)
            
            # Update index
            self._update_transformation_index(transformation_id, transformation)
            
            logger.info(f"Transformation recorded: {transformation_id}")
            return transformation_id
            
        except Exception as e:
            logger.error(f"Error recording transformation: {e}")
            raise
    
    def get_node_lineage(self, node_id: str) -> Dict[str, Any]:
        """Get lineage information for a node."""
        try:
            with open(self.transformation_index_file, 'r') as f:
                transform_index = json.load(f)
            
            lineage = {
                "node_id": node_id,
                "upstream": [],
                "downstream": [],
                "transformations": []
            }
            
            for trans_id, trans_info in transform_index["transformations"].items():
                if node_id in trans_info.get("input_nodes", []):
                    lineage["downstream"].append(trans_id)
                elif node_id in trans_info.get("output_nodes", []):
                    lineage["upstream"].append(trans_id)
                
                if (node_id in trans_info.get("input_nodes", []) or 
                    node_id in trans_info.get("output_nodes", [])):
                    lineage["transformations"].append({
                        "transformation_id": trans_id,
                        "operation_type": trans_info["operation_type"],
                        "execution_time": trans_info["execution_time"]
                    })
            
            return lineage
            
        except Exception as e:
            logger.error(f"Error getting node lineage: {e}")
            return {}
    
    def search_nodes(self, name_pattern: str = None, data_type: DataType = None) -> List[Dict[str, Any]]:
        """Search for data nodes."""
        try:
            with open(self.lineage_index_file, 'r') as f:
                lineage_index = json.load(f)
            
            matching_nodes = []
            
            for node_id, node_info in lineage_index["nodes"].items():
                node_file = self.lineage_path / "nodes" / f"{node_id}.json"
                if not node_file.exists():
                    continue
                
                with open(node_file, 'r') as f:
                    node_data = json.load(f)
                
                match = True
                if name_pattern and name_pattern.lower() not in node_data["name"].lower():
                    match = False
                if data_type and node_data["data_type"] != data_type.value:
                    match = False
                
                if match:
                    matching_nodes.append(node_data)
            
            return matching_nodes
            
        except Exception as e:
            logger.error(f"Error searching nodes: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata tracking statistics."""
        try:
            with open(self.lineage_index_file, 'r') as f:
                lineage_stats = json.load(f)["statistics"]
            
            with open(self.transformation_index_file, 'r') as f:
                transform_stats = json.load(f)["statistics"]
            
            return {
                "lineage": lineage_stats,
                "transformations": transform_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def _update_lineage_index(self, node_id: str, node: DataLineageNode):
        """Update lineage index."""
        try:
            with open(self.lineage_index_file, 'r') as f:
                index = json.load(f)
            
            index["nodes"][node_id] = {
                "name": node.name,
                "data_type": node.data_type.value,
                "location": node.location,
                "created_at": node.created_at,
                "size_bytes": node.size_bytes or 0
            }
            
            index["statistics"]["total_nodes"] = len(index["nodes"])
            
            data_type_key = node.data_type.value
            if data_type_key not in index["statistics"]["nodes_by_type"]:
                index["statistics"]["nodes_by_type"][data_type_key] = 0
            index["statistics"]["nodes_by_type"][data_type_key] += 1
            
            with open(self.lineage_index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating lineage index: {e}")
    
    def _update_transformation_index(self, transform_id: str, transformation: DataTransformation):
        """Update transformation index."""
        try:
            with open(self.transformation_index_file, 'r') as f:
                index = json.load(f)
            
            index["transformations"][transform_id] = {
                "operation_type": transformation.operation_type.value,
                "execution_time": transformation.execution_time,
                "duration_seconds": transformation.duration_seconds,
                "success": transformation.success,
                "input_nodes": transformation.input_nodes,
                "output_nodes": transformation.output_nodes
            }
            
            index["statistics"]["total_transformations"] = len(index["transformations"])
            
            op_type = transformation.operation_type.value
            if op_type not in index["statistics"]["operations_by_type"]:
                index["statistics"]["operations_by_type"][op_type] = 0
            index["statistics"]["operations_by_type"][op_type] += 1
            
            with open(self.transformation_index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating transformation index: {e}")


def create_metadata_tracker(base_path: Path) -> MetadataTracker:
    """Create a MetadataTracker instance."""
    return MetadataTracker(base_path) 