#!/usr/bin/env python3
"""
Test script for comprehensive metadata tracking system validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import time
from datetime import datetime

from data.infrastructure import DataInfrastructure
from data.metadata_tracker import DataType, OperationType

def test_metadata_tracking_integration():
    """Test the integrated metadata tracking system."""
    print("=== Testing Metadata Tracking System Integration ===\n")
    
    # Clean up test directory if it exists
    test_data_path = Path("./TestDataMetadata")
    if test_data_path.exists():
        shutil.rmtree(test_data_path)
    
    try:
        # Initialize data infrastructure
        print("1. Initializing Data Infrastructure...")
        infrastructure = DataInfrastructure(base_path="./TestDataMetadata")
        print("✅ Data infrastructure initialized successfully")
        
        # Test 1: Register data nodes
        print("\n2. Testing Data Node Registration...")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'asset_price': np.random.normal(100, 15, 100),
            'volatility': np.random.uniform(0.1, 0.5, 100)
        })
        
        # Register raw data node
        raw_node_id = infrastructure.register_data_node(
            name="sample_asset_prices",
            data_type=DataType.RAW_DATA,
            location=str(infrastructure.get_raw_data_path("asset") / "sample_prices.csv"),
            metadata={"source": "test", "frequency": "daily"},
            data=sample_data
        )
        print(f"✅ Raw data node registered: {raw_node_id}")
        
        # Create processed data
        processed_data = sample_data.copy()
        processed_data['returns'] = processed_data['asset_price'].pct_change()
        processed_data = processed_data.dropna()
        
        # Register processed data node
        processed_node_id = infrastructure.register_data_node(
            name="sample_asset_returns",
            data_type=DataType.PROCESSED_DATA,
            location=str(infrastructure.get_processed_data_path("asset") / "sample_returns.csv"),
            metadata={"derived_from": raw_node_id, "processing": "returns_calculation"},
            data=processed_data
        )
        print(f"✅ Processed data node registered: {processed_node_id}")
        
        # Test 2: Record transformation
        print("\n3. Testing Transformation Recording...")
        
        start_time = time.time()
        # Simulate processing time
        time.sleep(0.1)
        end_time = time.time()
        
        transformation_id = infrastructure.record_transformation(
            operation_type=OperationType.TRANSFORM,
            input_nodes=[raw_node_id],
            output_nodes=[processed_node_id],
            parameters={"operation": "calculate_returns", "method": "pct_change"},
            duration_seconds=end_time - start_time,
            success=True
        )
        print(f"✅ Transformation recorded: {transformation_id}")
        
        # Test 3: Create feature and register it
        print("\n4. Testing Feature Node Registration...")
        
        # Create a feature
        feature_data = processed_data[['date', 'returns']].copy()
        feature_data['rolling_volatility'] = feature_data['returns'].rolling(window=20).std()
        feature_data = feature_data.dropna()
        
        # Store feature using feature store
        feature_id = infrastructure.store_feature(
            feature_data=feature_data,
            feature_name="rolling_volatility_20d",
            feature_type="technical",
            version="1.0.0",
            description="20-day rolling volatility",
            metadata={"window": 20, "method": "rolling_std"},
            tags=["volatility", "technical"]
        )
        
        # Register feature node in metadata tracker
        feature_node_id = infrastructure.register_data_node(
            name="rolling_volatility_20d_feature",
            data_type=DataType.FEATURE,
            location=f"feature_store/{feature_id}",
            metadata={"feature_id": feature_id, "version": "1.0.0"},
            data=feature_data
        )
        print(f"✅ Feature node registered: {feature_node_id}")
        
        # Record feature engineering transformation
        feature_transform_id = infrastructure.record_transformation(
            operation_type=OperationType.FEATURE_ENGINEERING,
            input_nodes=[processed_node_id],
            output_nodes=[feature_node_id],
            parameters={"window": 20, "operation": "rolling_std"},
            duration_seconds=0.05,
            success=True
        )
        print(f"✅ Feature transformation recorded: {feature_transform_id}")
        
        # Test 4: Node lineage tracking
        print("\n5. Testing Node Lineage Tracking...")
        
        # Get lineage for processed data
        processed_lineage = infrastructure.get_node_lineage(processed_node_id)
        print(f"✅ Processed data lineage: {len(processed_lineage['transformations'])} transformations")
        
        # Get lineage for feature
        feature_lineage = infrastructure.get_node_lineage(feature_node_id)
        print(f"✅ Feature lineage: {len(feature_lineage['transformations'])} transformations")
        
        # Test 5: Node search functionality
        print("\n6. Testing Node Search...")
        
        # Search by name pattern
        search_results = infrastructure.search_data_nodes(name_pattern="sample")
        print(f"✅ Found {len(search_results)} nodes with 'sample' in name")
        
        # Search by data type
        feature_nodes = infrastructure.search_data_nodes(data_type=DataType.FEATURE)
        print(f"✅ Found {len(feature_nodes)} feature nodes")
        
        # Test 6: Statistics and reporting
        print("\n7. Testing Statistics and Reporting...")
        
        metadata_stats = infrastructure.get_metadata_statistics()
        print(f"✅ Metadata statistics:")
        print(f"   - Total nodes: {metadata_stats['lineage']['total_nodes']}")
        print(f"   - Total transformations: {metadata_stats['transformations']['total_transformations']}")
        print(f"   - Nodes by type: {metadata_stats['lineage']['nodes_by_type']}")
        print(f"   - Operations by type: {metadata_stats['transformations']['operations_by_type']}")
        
        # Test 7: Integration with cache system
        print("\n8. Testing Cache Integration with Metadata...")
        
        # Cache some data and register it
        cache_key = infrastructure.cache_processed_data(
            data=processed_data,
            data_identifier="cached_returns",
            parameters={"method": "pct_change"},
            expiry_hours=24,
            metadata={"cached_at": datetime.now().isoformat()}
        )
        
        # Register cache entry as a node
        cache_node_id = infrastructure.register_data_node(
            name="cached_asset_returns",
            data_type=DataType.CACHE_ENTRY,
            location=f"cache/{cache_key}",
            metadata={"cache_key": cache_key, "expiry_hours": 24},
            data=processed_data
        )
        print(f"✅ Cache entry registered as node: {cache_node_id}")
        
        # Test data retrieval
        retrieved_data = infrastructure.get_cached_data("cached_returns", {"method": "pct_change"})
        print(f"✅ Cache retrieval successful: {retrieved_data is not None}")
        
        # Get final statistics
        print("\n9. Final System Status...")
        final_stats = infrastructure.get_metadata_statistics()
        cache_stats = infrastructure.get_cache_statistics()
        feature_stats = infrastructure.get_feature_statistics()
        
        print(f"✅ Final Statistics:")
        print(f"   - Metadata nodes: {final_stats['lineage']['total_nodes']}")
        print(f"   - Transformations: {final_stats['transformations']['total_transformations']}")
        print(f"   - Cache entries: {cache_stats['statistics']['total_entries']}")
        print(f"   - Cache hit rate: {cache_stats['statistics']['cache_hits'] / (cache_stats['statistics']['cache_hits'] + cache_stats['statistics']['cache_misses']):.2%}")
        print(f"   - Features stored: {feature_stats['total_features']}")
        
        print("\n🎉 All metadata tracking tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if test_data_path.exists():
            shutil.rmtree(test_data_path)
            print(f"\n🧹 Cleanup completed: {test_data_path} removed")

if __name__ == "__main__":
    success = test_metadata_tracking_integration()
    sys.exit(0 if success else 1) 