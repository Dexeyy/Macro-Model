import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_dataframe(df, name):
    """
    Validate DataFrame for required properties.
    
    Args:
        df: DataFrame to validate
        name: Name of the DataFrame for logging
        
    Returns:
        Validated DataFrame
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            logger.info(f"Converted {name} index to DatetimeIndex")
        except Exception as e:
            raise ValueError(f"Could not convert {name} index to datetime: {e}")
    
    return df

def diagnose_dataframe(df, name):
    """
    Diagnose DataFrame issues.
    
    Args:
        df: DataFrame to diagnose
        name: Name of the DataFrame for logging
        
    Returns:
        DataFrame (unchanged)
    """
    logger.info(f"\nDiagnosing {name}:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Index type: {type(df.index)}")
    logger.info(f"Index range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    return df

def safe_divide(a, b, fill_value=0):
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        fill_value: Value to return if division by zero
        
    Returns:
        Result of division or fill_value if division by zero
    """
    try:
        return a / b if b != 0 else fill_value
    except Exception:
        return fill_value

def save_data(data, file_path, file_format='csv'):
    """
    Save data to file.
    
    Args:
        data: Data to save (DataFrame or dict)
        file_path: Path to save the file
        file_format: Format to save the file ('csv', 'json', 'pickle')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path)
                logger.info(f"Saved DataFrame to {file_path}")
                return True
            else:
                logger.error(f"Data is not a DataFrame, cannot save as CSV")
                return False
        
        elif file_format == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(file_path)
                logger.info(f"Saved DataFrame to {file_path}")
                return True
            elif isinstance(data, dict):
                with open(file_path, 'w') as f:
                    json.dump(data, f, default=str)
                logger.info(f"Saved dictionary to {file_path}")
                return True
            else:
                logger.error(f"Data type not supported for JSON")
                return False
        
        elif file_format == 'pickle':
            if isinstance(data, pd.DataFrame) or isinstance(data, dict):
                pd.to_pickle(data, file_path)
                logger.info(f"Saved data to {file_path}")
                return True
            else:
                logger.error(f"Data type not supported for pickle")
                return False
        
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return False
    
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        return False

def load_data(file_path, file_format='csv'):
    """
    Load data from file.
    
    Args:
        file_path: Path to load the file from
        file_format: Format of the file ('csv', 'json', 'pickle')
        
    Returns:
        Loaded data or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None
        
        if file_format == 'csv':
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded DataFrame from {file_path}")
            return data
        
        elif file_format == 'json':
            try:
                data = pd.read_json(file_path)
                logger.info(f"Loaded DataFrame from {file_path}")
                return data
            except:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded dictionary from {file_path}")
                return data
        
        elif file_format == 'pickle':
            data = pd.read_pickle(file_path)
            logger.info(f"Loaded data from {file_path}")
            return data
        
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return None
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def generate_timestamp():
    """
    Generate a timestamp string for file naming.
    
    Returns:
        String with timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_output_filename(base_name, extension, output_dir='output'):
    """
    Create an output filename with timestamp.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        output_dir: Output directory
        
    Returns:
        Full file path
    """
    timestamp = generate_timestamp()
    filename = f"{base_name}_{timestamp}.{extension}"
    return os.path.join(output_dir, filename) 