#!/usr/bin/env python3
"""Extract configuration parameters from opus-et config.pkl file."""

import pickle
import sys
import json


def extract_config(config_path):
    """Extract key parameters from config.pkl.
    
    Args:
        config_path: Path to config.pkl file
        
    Returns:
        dict with training Apix, original Apix (accounting for downfrac),
        D, particles_path, and effective_box_size
    """
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    lattice_D = config['lattice_args']['D']
    effective_box_size = lattice_D - 1  # Training uses D-1
    
    training_apix = config['model_args']['Apix']
    downfrac = config['dataset_args'].get('downfrac', 1.0)
    original_apix = training_apix * downfrac
    
    return {
        'training_Apix': training_apix,
        'downfrac': downfrac,
        'original_Apix': original_apix,
        'lattice_D': lattice_D,
        'effective_box_size': effective_box_size,
        'particles_path': config['dataset_args']['particles'],
        'poses_path': config['dataset_args']['poses'],
        'zdim': config['model_args']['zdim']
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: extract_config.py <config.pkl>", file=sys.stderr)
        print("\nOutputs key config values including:", file=sys.stderr)
        print("  training_Apix  - Pixel size used during training", file=sys.stderr)
        print("  downfrac       - Downsampling fraction applied during training", file=sys.stderr)
        print("  original_Apix  - Original data pixel size (training_Apix * downfrac)", file=sys.stderr)
        print("  effective_box_size - D-1 (use this for parse_pose_star -D)", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    params = extract_config(config_path)
    
    # Output as JSON for easy parsing
    print(json.dumps(params, indent=2))
