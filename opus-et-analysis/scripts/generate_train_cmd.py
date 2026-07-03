#!/usr/bin/env python3
"""
Generate cryo-ET multi-body training command for opus-et/cryoDRGN.

Usage:
    python generate_train_cmd.py [options]
    
    Interactive mode (no arguments):
        python generate_train_cmd.py
        
    Command-line mode:
        python generate_train_cmd.py \\
            --star ../zribo_test/matching80s.star \\
            --datadir /work/data/warp_tiltseries/ \\
            --mask-mrc ../zribo_test/mask.mrc \\
            --mask-params ../mask_params.pkl \\
            --tilt-range 50 --tilt-step 2 --angpix 3.37
"""

import argparse
import os
import struct
import sys


def get_star_column(star_file, column_name):
    """Find the column index for a given column name in star file."""
    with open(star_file, 'r') as f:
        for line in f:
            line = line.strip()
            if column_name in line:
                parts = line.split('#')
                if len(parts) > 1:
                    return int(parts[1].strip()) - 1  # 0-indexed
    return None


def get_first_subtomo_path(star_file, image_col):
    """Get the first subtomogram path from star file."""
    with open(star_file, 'r') as f:
        found_loop = False
        for line in f:
            line = line.strip()
            if line.startswith('loop_'):
                found_loop = True
                continue
            if found_loop and line and not line.startswith('_') and not line.startswith('data_'):
                parts = line.split()
                if len(parts) > image_col:
                    return parts[image_col]
    return None


def get_mrc_box_size(mrc_path):
    """Read MRC file header to get box size (NX dimension)."""
    try:
        with open(mrc_path, 'rb') as f:
            header = f.read(12)
            nx, ny, nz = struct.unpack('3i', header)
            return nx
    except Exception as e:
        print(f"Warning: Could not read MRC file {mrc_path}: {e}")
        return None


def get_subtomo_box_size(star_file, datadir):
    """Get the box size from the first subtomogram in star file."""
    # Find ImageName column
    image_col = get_star_column(star_file, '_rlnImageName')
    if image_col is None:
        print("Warning: Could not find _rlnImageName column in star file")
        return None
    
    # Get first subtomogram path
    first_image = get_first_subtomo_path(star_file, image_col)
    if first_image is None:
        print("Warning: Could not find first subtomogram in star file")
        return None
    
    # Resolve path relative to datadir
    full_path = os.path.join(datadir, first_image)
    
    # Get box size from MRC
    box_size = get_mrc_box_size(full_path)
    if box_size:
        print(f"Detected subtomogram box size: {box_size} (from {first_image})")
        return box_size
    
    return None


def get_args_interactive():
    """Get parameters interactively from user."""
    print("=" * 60)
    print("opus-et Training Command Generator")
    print("=" * 60)
    print()
    
    # Required experimental parameters
    print("--- Required Experimental Parameters ---")
    star_file = input("Star file path: ").strip()
    
    # Need datadir first to auto-detect box size
    datadir = input("Tilt series directory: ").strip()
    
    # Auto-detect box size from subtomograms
    if star_file and datadir:
        detected_box = get_subtomo_box_size(star_file, datadir)
        if detected_box:
            box_size_input = input(f"Box size for pose generation [{detected_box}]: ").strip()
            box_size = box_size_input if box_size_input else str(detected_box)
        else:
            box_size = input("Box size for pose generation: ").strip()
    else:
        box_size = input("Box size for pose generation: ").strip()
    
    poses = input("Pose pickle path (or empty to generate from star file): ").strip()
    mask_mrc = input("Mask MRC file path: ").strip()
    tilt_range = input("Tilt range (degrees) [50]: ").strip() or "50"
    tilt_step = input("Tilt step (degrees) [2]: ").strip() or "2"
    angpix = input("Pixel size (Angstroms) [3.37]: ").strip() or "3.37"
    
    # Multi-body parameters
    print()
    print("--- Multi-Body Parameters (optional) ---")
    mask_params = input("Mask params path (mask_params.pkl, or empty): ").strip()
    split = input("Split pickle path [deep.pkl]: ").strip() or "deep.pkl"
    
    # Optional parameters
    print()
    print("--- Optional Parameters ---")
    zdim = input("ZDIM (composition latent dim) [12]: ").strip() or "12"
    zaffinedim = input("ZAFFINEDIM (conformation latent dim) [4]: ").strip() or "4"
    encoderres = input("ENCODERRES (encoder resolution, or empty for default): ").strip()
    num_epochs = input("Number of epochs [40]: ").strip() or "40"
    batch_size = input("Batch size [10]: ").strip() or "10"
    learning_rate = input("Learning rate [3.0e-5]: ").strip() or "3.0e-5"
    beta_control = input("Beta control (KL weight) [0.5]: ").strip() or "0.5"
    lamb = input("Lambda (disentanglement) [0.5]: ").strip() or "0.5"
    bfactor = input("B-factor [3.0]: ").strip() or "3.0"
    num_gpus = input("Number of GPUs [4]: ").strip() or "4"
    output_dir = input("Output directory [.]: ").strip() or "."
    val_frac = input("Validation fraction [0.05]: ").strip() or "0.05"
    templateres = input("Template resolution (box size) [128]: ").strip() or "128"
    
    return {
        'star_file': star_file,
        'poses': poses,
        'datadir': datadir,
        'mask_mrc': mask_mrc,
        'mask_params': mask_params,
        'split': split,
        'tilt_range': tilt_range,
        'tilt_step': tilt_step,
        'angpix': angpix,
        'box_size': box_size,
        'zdim': zdim,
        'zaffinedim': zaffinedim,
        'encoderres': encoderres,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'beta_control': beta_control,
        'lamb': lamb,
        'bfactor': bfactor,
        'num_gpus': num_gpus,
        'output_dir': output_dir,
        'val_frac': val_frac,
        'templateres': templateres,
    }


def get_args_cli():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(
        description='Generate cryo-ET training command for opus-et',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python generate_train_cmd.py
    
  Command-line mode (auto-detect box size):
    python generate_train_cmd.py --star particles.star \\
        --datadir /path/to/tiltseries/ --mask-mrc mask.mrc \\
        --tilt-range 50 --tilt-step 2 --angpix 3.37
        
  Command-line mode (with existing poses):
    python generate_train_cmd.py --star particles.star --poses poses.pkl \\
        --datadir /path/to/tiltseries/ --mask-mrc mask.mrc \\
        --tilt-range 50 --tilt-step 2 --angpix 3.37
        """
    )
    
    # Required arguments
    parser.add_argument('--star', required=True, help='Particle star file path')
    parser.add_argument('--datadir', required=True, help='Tilt series directory path')
    parser.add_argument('--mask-mrc', required=True, help='Mask MRC file path')
    parser.add_argument('--tilt-range', type=float, required=True, help='Maximum tilt angle (degrees)')
    parser.add_argument('--tilt-step', type=float, required=True, help='Tilt increment (degrees)')
    parser.add_argument('--angpix', type=float, required=True, help='Pixel size (Angstroms)')
    
    # Optional: provide box size manually (otherwise auto-detect from subtomograms)
    parser.add_argument('--box-size', type=int, default=None, help='Box size for pose generation (auto-detected if not provided)')
    
    # Multi-body arguments
    parser.add_argument('--poses', default='', help='Pose pickle file path (optional, will generate from star if not provided)')
    parser.add_argument('--mask-params', default='', help='Path to mask_params.pkl for multi-body (optional)')
    parser.add_argument('--split', default='deep.pkl', help='Train/val split pickle [deep.pkl]')
    parser.add_argument('--no-estpose', action='store_true', help='Skip pose estimation (if poses already refined)')
    
    # Optional training parameters
    parser.add_argument('--zdim', type=int, default=12, help='Composition latent dimension [12]')
    parser.add_argument('--zaffinedim', type=int, default=4, help='Conformation latent dimension [4]')
    parser.add_argument('--encoderres', type=int, default=None, help='Encoder resolution (optional)')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of epochs [40]')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size per GPU [10]')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate [3.0e-5]')
    parser.add_argument('--beta', type=float, default=0.5, help='KL divergence weight [0.5]')
    parser.add_argument('--lamb', type=float, default=0.5, help='Disentanglement weight [0.5]')
    parser.add_argument('--bfactor', type=float, default=3.0, help='B-factor sharpening [3.0]')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs [4]')
    parser.add_argument('--output-dir', default='.', help='Output directory [.]')
    parser.add_argument('--val-frac', type=float, default=0.05, help='Validation fraction [0.05]')
    parser.add_argument('--templateres', type=int, default=128, help='Output box size [128]')
    parser.add_argument('--output', '-o', help='Output script file (optional)')
    
    args = parser.parse_args()
    
    # Auto-detect box size if not provided
    box_size = args.box_size
    if box_size is None:
        detected = get_subtomo_box_size(args.star, args.datadir)
        if detected:
            box_size = detected
        else:
            print("Error: Could not auto-detect box size. Please provide --box-size")
            sys.exit(1)
    
    return {
        'star_file': args.star,
        'poses': args.poses,
        'datadir': args.datadir,
        'mask_mrc': args.mask_mrc,
        'mask_params': args.mask_params,
        'split': args.split,
        'tilt_range': str(args.tilt_range),
        'tilt_step': str(args.tilt_step),
        'angpix': str(args.angpix),
        'box_size': str(box_size),
        'zdim': str(args.zdim),
        'zaffinedim': str(args.zaffinedim),
        'encoderres': str(args.encoderres) if args.encoderres else '',
        'num_epochs': str(args.num_epochs),
        'batch_size': str(args.batch_size),
        'learning_rate': str(args.lr),
        'beta_control': str(args.beta),
        'lamb': str(args.lamb),
        'bfactor': str(args.bfactor),
        'num_gpus': str(args.num_gpus),
        'output_dir': args.output_dir,
        'val_frac': str(args.val_frac),
        'templateres': str(args.templateres),
        'output_file': getattr(args, 'output', None),
        'estpose': not args.no_estpose,
    }


def generate_script(params):
    """Generate bash training script."""
    
    # Build conditional arguments with newlines
    masks_arg = "    --masks ${MASK_PARAMS} \\\n" if params['mask_params'] else ""
    mask_params_line = f"MASK_PARAMS={params['mask_params']}" if params['mask_params'] else "# MASK_PARAMS=<path_to_mask_params>  # Uncomment for multi-body"
    
    encoderres_var_line = f"ENCODERRES={params['encoderres']}" if params['encoderres'] else "# ENCODERRES=<encoder_res>  # Optional"
    encoderres_cmd_line = "    --encoderres ${ENCODERRES} \\\n" if params['encoderres'] else ""
    
    # Handle pose pkl generation
    if params['poses']:
        # User provided pose pkl
        pose_pkl_var = f"POSE_PKL={params['poses']}"
        pose_gen_section = ""
    else:
        # Generate pose pkl from star file
        star_base = os.path.splitext(os.path.basename(params['star_file']))[0]
        pose_pkl_var = f"POSE_PKL={star_base}_pose.pkl"
        # Use detected box size
        box_size = params['box_size']
        pose_gen_section = f'''# ==============================================================================
# GENERATE POSE PKL FROM STAR FILE
# ==============================================================================

echo "Generating pose pickle from star file..."
dsd parse_pose_star ${{STAR_FILE}} \\
    -D {box_size} \\
    --Apix ${{ANGPIX}} \\
    -o ${{POSE_PKL}}

echo "Pose pickle saved to: ${{POSE_PKL}}"

'''
    
    # Handle estpose flag
    estpose_line = "    --estpose" if params.get('estpose', True) else "#    --estpose  # Pose refinement disabled"
    
    script = f'''#!/bin/bash

# ==============================================================================
# EXPERIMENTAL PARAMETERS
# ==============================================================================

STAR_FILE={params['star_file']}
{pose_pkl_var}
DATADIR={params['datadir']}
MASK_MRC={params['mask_mrc']}
TILT_RANGE={params['tilt_range']}
TILT_STEP={params['tilt_step']}
ANGPIX={params['angpix']}

# Multi-body deformation (optional)
{mask_params_line}
SPLIT_PKL={params['split']}

# ==============================================================================
# OPTIONAL/TUNABLE PARAMETERS
# ==============================================================================

ZDIM={params['zdim']}
ZAFFINEDIM={params['zaffinedim']}
{encoderres_var_line}
NUM_EPOCHS={params['num_epochs']}
BATCH_SIZE={params['batch_size']}
LEARNING_RATE={params['learning_rate']}
BETA_CONTROL={params['beta_control']}
LAMB={params['lamb']}
BFACTOR={params['bfactor']}
NUM_GPUS={params['num_gpus']}
OUTPUT_DIR={params['output_dir']}
VAL_FRAC={params['val_frac']}
TEMPLATERES={params['templateres']}

{pose_gen_section}# ==============================================================================
# TRAINING COMMAND
# ==============================================================================

torchrun --nproc_per_node=${{NUM_GPUS}} -m cryodrgn.commands.train_tomo_dist \\
    ${{STAR_FILE}} \\
    --poses ${{POSE_PKL}} \\
    -n ${{NUM_EPOCHS}} \\
    -b ${{BATCH_SIZE}} \\
    --zdim ${{ZDIM}} \\
    --zaffinedim ${{ZAFFINEDIM}} \\
{encoderres_cmd_line}    --lr ${{LEARNING_RATE}} \\
    --num-gpus ${{NUM_GPUS}} \\
    --multigpu \\
    --beta-control ${{BETA_CONTROL}} \\
    -o ${{OUTPUT_DIR}} \\
    -r ${{MASK_MRC}} \\
{masks_arg}    --split ${{SPLIT_PKL}} \\
    --lamb ${{LAMB}} \\
    --bfactor ${{BFACTOR}} \\
    --valfrac ${{VAL_FRAC}} \\
    --templateres ${{TEMPLATERES}} \\
    --tmp-prefix tmp \\
    --datadir ${{DATADIR}} \\
    --angpix ${{ANGPIX}} \\
    --downfrac 1. \\
    --warp \\
    --tilt-range ${{TILT_RANGE}} \\
    --tilt-step ${{TILT_STEP}} \\
    --ctfalpha 0. \\
    --ctfbeta 1. \\
{estpose_line}
'''
    return script


def main():
    # Check if running interactively or with arguments
    if len(sys.argv) == 1:
        # Interactive mode
        params = get_args_interactive()
        output_file = input("\nOutput script filename [train.sh]: ").strip() or "train.sh"
    else:
        # Command-line mode
        params = get_args_cli()
        output_file = params.pop('output_file', None) or "train.sh"
    
    # Generate script
    script = generate_script(params)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(script)
    
    # Make executable
    os.chmod(output_file, 0o755)
    
    print(f"\n{'='*60}")
    print(f"Training script generated: {output_file}")
    print(f"{'='*60}")
    print(f"\nRun with: ./{output_file}")
    
    # Also print the command for reference
    print("\n--- Generated Script Preview ---")
    print(script)


if __name__ == '__main__':
    main()
