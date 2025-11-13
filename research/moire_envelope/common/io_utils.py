"""
I/O utilities for managing runs and outputs
"""
from pathlib import Path
import json
import yaml
from datetime import datetime
import numpy as np


def ensure_run_dir(config):
    """
    Create a timestamped run directory
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path: Run directory path
    """
    base_dir = Path(config.get('output_dir', 'runs'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get('run_name', 'run')
    run_dir = base_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to run directory
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return run_dir


def candidate_dir(run_dir, candidate_id):
    """
    Get directory path for a specific candidate
    
    Args:
        run_dir: Run directory path
        candidate_id: Candidate ID number
        
    Returns:
        Path: Candidate directory path
    """
    cdir = Path(run_dir) / f"candidate_{candidate_id:04d}"
    return cdir


def load_yaml(path):
    """
    Load YAML configuration file
    
    Args:
        path: Path to YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data, path):
    """
    Save data as JSON
    
    Args:
        data: Data to save
        path: Output path
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """
    Load JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        dict: Loaded data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def choose_reference_frequency(omega0_grid, config):
    """
    Choose reference frequency from omega0 grid
    
    Args:
        omega0_grid: 2D array of frequencies
        config: Configuration with 'ref_frequency_mode'
        
    Returns:
        float: Reference frequency
    """
    mode = config.get('ref_frequency_mode', 'mean')
    
    if mode == 'mean':
        return float(omega0_grid.mean())
    elif mode == 'min':
        return float(omega0_grid.min())
    elif mode == 'max':
        return float(omega0_grid.max())
    elif mode == 'median':
        return float(np.median(omega0_grid))
    else:
        return float(omega0_grid.mean())
