---
name: opus-et-analysis
description: Cryo-ET data processing and analysis workflows using opus-et training results. Handles PCA/kmeans clustering, volume generation from latent codes, pose parsing, and star file manipulation. Use when processing training results from a specific epoch, generating volumes for cluster centers or principal components, parsing poses, or combining star files from cryo-ET reconstructions.
---

# opus-et Analysis

Process and analyze cryo-ET training results from opus-et.

## Agent Rules — read before acting

- **Do only what the user asks.** Don't anticipate, extend, or fix unreported issues — even obvious ones. One change at a time.
- **Read before editing.** Always read the current file/script before describing or modifying it. Don't rely on remembered content — the codebase changes.
- **Verify before asserting.** If uncertain how a tool, flag, or script behaves, check (`--help`, API docs, actual output) rather than inferring from naming. Wrong documentation is worse than no documentation.
- **Enumerate scope before acting.** For multi-file changes, use `grep` to find all affected files and confirm scope with the user before making any edits.
- **Show before and after for every edit.** Before modifying a script or config, quote the relevant current lines. After editing, summarize exactly what changed — not just "done."
- **Don't chain changes.** Renaming a variable in one place does NOT mean you should rename it everywhere — confirm scope first.
- **Don't redesign.** Apparent inconsistencies may be intentional. Respect the existing pattern unless the user asks to change it.
- **When in doubt, show the current state and ask.** "Here's what I see. Do you want me to change X, Y, or both?"


## Training Command Generation

Use the provided script to generate training commands:

```bash
python <skill-dir>/scripts/generate_train_cmd.py [options]
```

### Pose PKL Generation

The pose pickle file can be provided directly or **generated automatically** from the star file:
- If `--poses` is provided, it will be used directly
- If `--poses` is omitted, the script will:
  1. Parse the star file to find the first subtomogram
  2. Read the MRC header to detect the actual box size
  3. Generate pose pkl using `dsd parse_pose_star` with the correct dimensions

### Interactive Mode

Run without arguments to be prompted for all parameters:

```bash
python <skill-dir>/scripts/generate_train_cmd.py
```

### Command-Line Mode

**With existing pose pkl:**
```bash
python <skill-dir>/scripts/generate_train_cmd.py \
    --star ../zribo_test/matching80s.star \
    --poses ../zribo_test/matching80s_pose_euler.pkl \
    --datadir /work/jpma/luo/tomo/warp_DEF/metadata/warp_tiltseries/ \
    --mask-mrc ../zribo_test/mask.mrc \
    --mask-params ../mask_params.pkl \
    --split deep.pkl \
    --tilt-range 50 \
    --tilt-step 2 \
    --angpix 3.37 \
    --output train.sh
```

**Auto-generate pose pkl from star file:**
```bash
python <skill-dir>/scripts/generate_train_cmd.py \
    --star ../zribo_test/matching80s.star \
    --datadir /work/jpma/luo/tomo/warp_DEF/metadata/warp_tiltseries/ \
    --mask-mrc ../zribo_test/mask.mrc \
    --mask-params ../mask_params.pkl \
    --split deep.pkl \
    --tilt-range 50 \
    --tilt-step 2 \
    --angpix 3.37 \
    --output train.sh
```

The generated script will auto-detect the subtomogram box size and include a section to create the pose pkl:
```bash
# ==============================================================================
# GENERATE POSE PKL FROM STAR FILE
# ==============================================================================

echo "Generating pose pickle from star file..."
dsd parse_pose_star ${STAR_FILE} \
    -D 128 \
    --Apix ${ANGPIX} \
    -o ${POSE_PKL}
```

**Note:** The box size (128 in this example) is automatically detected by reading the first subtomogram's MRC header. You can override this with `--box-size` if needed.

### Training Command Template

If writing manually, use this template. Note: You can generate `POSE_PKL` from `STAR_FILE` using `dsd parse_pose_star`.

```bash
#!/bin/bash

# ==============================================================================
# EXPERIMENTAL PARAMETERS
# ==============================================================================

STAR_FILE=<path_to_star>           # Particle star file
POSE_PKL=<path_to_pose_pkl>        # Pose pickle file (or generate from star)
DATADIR=<path_to_tilt_series>      # Path to tilt series directory
MASK_MRC=<path_to_mask_mrc>        # Mask mrc file
TILT_RANGE=<tilt_range>            # Maximum tilt angle (degrees)
TILT_STEP=<tilt_step>              # Tilt increment (degrees)
ANGPIX=<pixel_size>                # Pixel size in Angstroms

# Multi-body deformation (optional)
MASK_PARAMS=<path_to_mask_params>  # Path to mask_params.pkl for multi-body
SPLIT_PKL=<path_to_split>          # Train/val split pickle

# ==============================================================================
# OPTIONAL/TUNABLE PARAMETERS
# ==============================================================================

ZDIM=12                            # Composition latent space dimension
ZAFFINEDIM=4                       # Conformation latent space dimension
# ENCODERRES=13                    # Optional: encoder resolution
NUM_EPOCHS=40
BATCH_SIZE=10
LEARNING_RATE=3.0e-5
BETA_CONTROL=0.5                   # KL divergence weight
LAMB=0.5                           # Structural disentanglement weight
BFACTOR=3.0                        # B-factor sharpening
NUM_GPUS=4
OUTPUT_DIR=.
VAL_FRAC=0.05
TEMPLATERES=128

# ==============================================================================
# TRAINING COMMAND
# ==============================================================================

torchrun --nproc_per_node=${NUM_GPUS} -m cryodrgn.commands.train_tomo_dist \
    ${STAR_FILE} \
    --poses ${POSE_PKL} \
    -n ${NUM_EPOCHS} \
    -b ${BATCH_SIZE} \
    --zdim ${ZDIM} \
    --zaffinedim ${ZAFFINEDIM} \
    --lr ${LEARNING_RATE} \
    --num-gpus ${NUM_GPUS} \
    --multigpu \
    --beta-control ${BETA_CONTROL} \
    -o ${OUTPUT_DIR} \
    -r ${MASK_MRC} \
    --masks ${MASK_PARAMS} \
    --split ${SPLIT_PKL} \
    --lamb ${LAMB} \
    --bfactor ${BFACTOR} \
    --valfrac ${VAL_FRAC} \
    --templateres ${TEMPLATERES} \
    --tmp-prefix tmp \
    --datadir ${DATADIR} \
    --angpix ${ANGPIX} \
    --downfrac 1. \
    --warp \
    --tilt-range ${TILT_RANGE} \
    --tilt-step ${TILT_STEP} \
    --ctfalpha 0. \
    --ctfbeta 1. \
    --estpose
```

**Example values:**
| Parameter | Example Value |
|-----------|---------------|
| `STAR_FILE` | `../zribo_test/matching80s.star` |
| `POSE_PKL` | `../zribo_test/matching80s_pose_euler.pkl` |
| `DATADIR` | `/work/jpma/luo/tomo/warp_DEF/metadata/warp_tiltseries/` |
| `MASK_MRC` | `../zribo_test/mask.mrc` |
| `MASK_PARAMS` | `../mask_params.pkl` |
| `SPLIT_PKL` | `deep.pkl` |
| `TILT_RANGE` | `50` |
| `TILT_STEP` | `2` |
| `ANGPIX` | `3.37` |

### Quick Parameter Reference

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `DATADIR` | Required | Tilt series directory path | `/path/to/tilt_series/` |
| `ANGPIX` | Required | Pixel size (Å) | `3.37` |
| `TILT_STEP` | Required | Tilt increment (°) | `2` |
| `TILT_RANGE` | Required | Max tilt angle (°) | `50` |
| `--box-size` | Optional | Subtomogram box size (auto-detected from MRC) | `128` |
| `ZDIM` | Tunable | Composition latent dim | `12` |
| `--zaffinedim` | Tunable | Conformation latent dim | `4` (continuous conformational changes) |
| `BETA_CONTROL` | Tunable | Reconstruction vs KL balance | `0.5-1.0` |
| `LAMB` | Tunable | Disentanglement strength | `0.5-1.0` |
| `BFACTOR` | Tunable | Map sharpening factor | `3.0` |
| `TEMPLATERES` | Tunable | Output box size | `128` |
| `--warp` | Flag | Enable I/O for WarpTools subtomograms | Add when using WarpTools |
| `--masks` | Optional | Path to mask_params.pkl for multi-body deformation | `../mask_params.pkl` |

## Helper Scripts

The following helper scripts ship with this skill and are located in the skill's `scripts/` directory (not the project directory):

```
<skill-dir>/scripts/
```

`<skill-dir>` is the directory where this skill is installed. When invoking the scripts from a project working directory, replace `<skill-dir>` with the actual skill path, or copy/symlink the scripts into your project. Examples below use `<skill-dir>` as a placeholder.

### `generate_train_cmd.py`
Generates training commands with automatic pose pkl generation.
```bash
python <skill-dir>/scripts/generate_train_cmd.py [options]
```

### `extract_config.py`
Extracts configuration parameters from config.pkl.
```bash
python <skill-dir>/scripts/extract_config.py config.pkl
```

### `exclude_stars.py`
Checks overlap between two star files based on 3D coordinates.
```bash
python <skill-dir>/scripts/exclude_stars.py <reference.star> <query.star>
```

**Pixel size:** Auto-detected from `config.pkl` in the current directory (`Apix * downfrac`). Falls back to 3.37 Å if no config.pkl is found.

**Distance threshold:** `136/angpix` voxels = **136 Å** (constant physical distance regardless of pixel size).

## Determining the Original Pixel Size

The original data pixel size (needed for `parse_pose_star`, `exclude_stars.py`, etc.) can be found two ways:

**From config.pkl** (recommended):
```python
import pickle
config = pickle.load(open('config.pkl', 'rb'))
original_Apix = config['model_args']['Apix'] * config['dataset_args']['downfrac']
```

**From the star file** (alternative):
Check `_rlnDetectorPixelSize` (column 9) in the star file header — this is the original pixel size of the raw data.

The two values should agree closely. The star file value is the authoritative original; the config.pkl derivation is the training pipeline's record of it.

Use `<skill-dir>/scripts/extract_config.py` to extract config values including `original_Apix`.

## Quick Reference

Key parameters from `config.pkl`:
- `Apix = config['model_args']['Apix']` (training-effective pixel size)
- `downfrac = config['dataset_args']['downfrac']` (downsampling fraction)
- `original_Apix = Apix * downfrac` (original data pixel size — use for `parse_pose_star` and other original-data operations)
- `D = config['lattice_args']['D'] - 1` (effective box size is lattice D minus 1)
- `particles = config['dataset_args']['particles']` (original star file)

**Important:** The `Apix` stored in `config.pkl` is the training-effective pixel size. For operations on the original star file (e.g., `dsd parse_pose_star`), multiply by `downfrac` to get the original data pixel size:
```python
import pickle
config = pickle.load(open('config.pkl', 'rb'))
apix = config['model_args']['Apix'] * config['dataset_args']['downfrac']
```

Use `<skill-dir>/scripts/extract_config.py` to extract these values.

## Multi-Body Training

To enable multi-body deformation modeling, add `--masks <path_to_mask_params.pkl>` to the training command. The `mask_params.pkl` file contains rigid body definitions:

| Key | Description |
|-----|-------------|
| `com_bodies` | Centers of mass for each rigid body (shape: `[num_bodies, 3]`) |
| `principal_axes` | Principal axes defining body orientations |
| `orient_bodies` | Body orientation matrices |
| `rotate_directions` | Allowed rotation directions for each body |
| `in_relatives` | Rotation reference body index - body *i* rotates relative to body `in_relatives[i]` |
| `radii_bodies` | Radii for each body |

### Parameter Clarification

- **`ZDIM`**: Composition latent space dimension - captures structural/compositional heterogeneity
- **`--zaffinedim`**: Conformation latent space dimension - captures continuous conformational changes (independent of deformation modeling)
- **`--masks`**: Enables rigid body deformation modeling using the body definitions in `mask_params.pkl`

These three mechanisms operate independently:
- **Composition** (`ZDIM`): Discrete structural states
- **Conformation** (`--zaffinedim`): Continuous flexible motions  
- **Deformation** (`--masks`): Rigid body motions between defined bodies

## Common Workflows

### 1. Analyze Epoch (PCA + K-means)

Run PCA and kmeans clustering on a specific epoch:

```bash
dsdsh analyze <workdir> <epoch> <numpc> <numk>
```

Example:
```bash
dsdsh analyze . 39 10 20
```

Output: `analyze.39/` with `kmeans20/`, `pc1/` to `pc10/`, plots.

### 2. Generate Volumes for K-means Centers

```bash
dsd eval_vol --load weights.<epoch>.pkl \
    -c config.pkl \
    -o kmeans_volumes \
    --zfile analyze.<epoch>/kmeans<numk>/centers.txt \
    --Apix <apix> \
    --prefix kmeans_cluster
```

### 3. Generate Volumes for Principal Components

```bash
dsd eval_vol --load weights.<epoch>.pkl \
    -c config.pkl \
    -o pc_volumes/pc<N> \
    --zfile analyze.<epoch>/pc<N>/z_pc.txt \
    --Apix <apix> \
    --prefix pc<N>
```

### 4. Create Star Files for Clusters

Parse poses and split by kmeans cluster labels:

```bash
# First extract config to get correct D and Apix values
python <skill-dir>/scripts/extract_config.py config.pkl

# Then parse with correct box size (D-1 from lattice_args)
# Use original pixel size: config['model_args']['Apix'] * config['dataset_args']['downfrac']
dsd parse_pose_star <particles.star> \
    -D <effective_box_size> \
    --Apix <original_apix> \
    --labels analyze.<epoch>/kmeans<numk>/labels.pkl \
    --outdir <outdir>
```

**Use specific epoch poses:** To use poses from a specific epoch (e.g., `pose.29.pkl`) instead of the original star file poses:

```bash
dsd parse_pose_star <particles.star> \
    -D <effective_box_size> \
    --Apix <original_apix> \
    --poses pose.<epoch>.pkl \
    --labels analyze.<epoch>/kmeans<numk>/labels.pkl \
    --outdir <outdir>
```

**Critical:** 
- The effective box size is `lattice_args['D'] - 1`, not the raw D value.
- The original pixel size is `config['model_args']['Apix'] * config['dataset_args']['downfrac']`. Do not use the raw `config['model_args']['Apix']` for `parse_pose_star` — that is the training-effective pixel size.

### 5. Combine Star Files

Merge multiple cluster star files:

```bash
# Two files
dsdsh combine_star pre9.star pre10.star combined.star

# Multiple files (chain commands)
dsdsh combine_star pre9.star pre10.star temp1.star
dsdsh combine_star temp1.star pre11.star temp2.star
dsdsh combine_star temp2.star pre12.star combined_9_10_11_12.star
```

### 6. Generate Pose Pickle for Combined/Any Star File

Convert a star file to pose pickle format:

```bash
dsd parse_pose_star <starfile> \
    -D <effective_box_size> \
    --Apix <original_apix> \
    -o <output_pose.pkl>
```

Example for combined clusters:
```bash
dsd parse_pose_star kmeans_pose/combined_9_10_11_12.star \
    -D <effective_box_size> \
    --Apix <original_apix> \
    -o kmeans_pose/combined_9_10_11_12_pose.pkl
```

### 7. Check Overlap Between Star Files

Use the skill's `exclude_stars.py` to check overlap between two star files based on 3D coordinates (within 136 Å threshold):

```bash
python <skill-dir>/scripts/exclude_stars.py <reference.star> <query.star>
```

Pixel size is auto-detected from `config.pkl` in the current directory.

Output:
- Prints overlap statistics for each micrograph
- Generates `<query>_exclude.star` with non-overlapping particles

Example workflow to check overlap of all cluster star files with a test set:
```bash
for f in kmeans_pose/pre*.star; do
    echo "=== Checking overlap for $f ==="
    python <skill-dir>/scripts/exclude_stars.py test_set.star "$f"
done
```

## Complete Workflow Example

Full pipeline from analysis to combined pose generation:

```bash
# 0. Extract config values (original_Apix, effective_box_size)
python <skill-dir>/scripts/extract_config.py config.pkl

# 1. Analyze epoch
dsdsh analyze . 39 10 20

# 2. Generate volumes for kmeans centers
dsd eval_vol --load weights.39.pkl -c config.pkl -o kmeans_volumes \
    --zfile analyze.39/kmeans20/centers.txt --Apix <original_apix> --prefix kmeans_cluster

# 3. Create star files for all clusters
dsd parse_pose_star <particles.star> -D <effective_box_size> --Apix <original_apix> \
    --labels analyze.39/kmeans20/labels.pkl --outdir kmeans_pose

# 4. Combine specific clusters
dsdsh combine_star kmeans_pose/pre9.star kmeans_pose/pre10.star temp.star
dsdsh combine_star temp.star kmeans_pose/pre11.star temp2.star
dsdsh combine_star temp2.star kmeans_pose/pre12.star \
    kmeans_pose/combined_9_10_11_12.star

# 5. Generate pose pickle for combined clusters
dsd parse_pose_star kmeans_pose/combined_9_10_11_12.star \
    -D <effective_box_size> --Apix <original_apix> -o kmeans_pose/combined_9_10_11_12_pose.pkl
```

## Directory Structure Convention

After analysis:
```
.
├── analyze.<epoch>/
│   ├── kmeans<numk>/
│   │   ├── centers.txt      # Latent codes for cluster centers
│   │   ├── centers.pkl      # Numpy array of centers
│   │   ├── labels.pkl       # Cluster assignment for each particle
│   │   ├── centers_ind.txt  # Particle indices closest to each center
│   │   └── pre<N>.star      # Star files per cluster
│   ├── pc<N>/
│   │   └── z_pc.txt         # Latent codes along PC trajectory
│   └── *.png                # Visualization plots
├── kmeans_volumes/          # Generated cluster center volumes
├── pc_volumes/              # Generated PC trajectory volumes
│   ├── pc1/
│   ├── pc2/
│   └── ...
└── kmeans_pose/             # Star files split by cluster
```

### 8. Deformation Analysis

When the model is trained with deformation/warp parameters (e.g., for rigid body motion), the `analyze` command outputs both conformation latent space (`analyze.<epoch>/`) and deformation latent space (`defanalyze.<epoch>/`) results in one shot:

```bash
dsdsh analyze <workdir> <epoch> <numpc> <numk>
```

Example:
```bash
dsdsh analyze . 39 10 30
```

Output:
- `analyze.39/` - Full conformation space (zdim-dimensional, e.g., 12-dim)
- `defanalyze.39/` - Deformation parameter space (config-dependent dimensions, e.g., 4-dim for 2-body deformation)

Both directories contain similar structures (`kmeans<numk>/`, `pc<N>/`, plots).

### 9. Generate Deformation Volumes Along PCs

Generate volumes with rigid body deformation along principal components:

```bash
# Step 1: Create template z-file from k-means cluster
cat > template_z17.txt << 'EOF'
2.718417 1.193066 1.568582 0.121429 0.286391 -4.490979 0.128237 0.110625 -0.278317 1.854208 -1.253814 0.234952
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
EOF

# Step 2: Generate deformation volumes
dsd eval_vol --load weights.<epoch>.pkl \
    -c config.pkl \
    -o defanalyze_volumes/pc<N> \
    --deform \
    --masks <path_to_mask_params.pkl> \
    --template-z template_z17.txt \
    --template-z-ind 0 \
    --zfile defanalyze.<epoch>/pc<N>/z_pc.txt \
    --Apix <apix> \
    --prefix reference
```

**Key parameters:**
- `--deform`: Enable deformation mode
- `--masks`: Path to `mask_params.pkl` containing rigid body definitions
- `--template-z`: Text file with base conformation z-values (N-dim from analyze, 2D format: rows × zdim)
- `--template-z-ind`: Index of template to use (0 for first row)
- `defanalyze.<epoch>/pc<N>/z_pc.txt`: Deformation parameters (M-dimensional, from defanalyze)

**Note on dimensions:** The template z-values and deformation z-values have different dimensions:
- Template (from analyze): Matches `config['model_args']['zdim']` (check with `extract_config.py`)
- Deformation (from defanalyze): Matches number of deformation parameters (typically num_bodies × 2 for rotation+translation)

### 10. Create Template from K-means Cluster

Extract a k-means center as template for deformation analysis:

```python
import pickle
import numpy as np

# Load from analyze (non-deformation) results
centers = pickle.load(open('analyze.<epoch>/kmeans<numk>/centers.pkl', 'rb'))
center_17 = centers[17]

# Save as 2D array (required format for --template-z)
np.savetxt('template_z17.txt', center_17.reshape(1, -1), fmt='%.6f')
```

**Important:** The template-z file must be 2D (rows × zdim). For a single template, save as (1, zdim) array where zdim matches your model configuration.

### 11. Analyze Mask Parameters

Inspect rigid body definitions in `mask_params.pkl`:

```python
import torch

m = torch.load('mask_params.pkl', map_location='cpu')
print('Keys:', list(m.keys()))
# Output: ['in_relatives', 'com_bodies', 'orient_bodies', 
#          'rotate_directions', 'radii_bodies', 'principal_axes']

# Check number of bodies
print('Number of bodies:', m['com_bodies'].shape[0])
print('COM of bodies:', m['com_bodies'])
print('Principal axes:', m['principal_axes'])
```

## Complete Deformation Workflow Example

Full pipeline for generating deformation volumes along PCs:

```bash
# 0. Extract config values
python <skill-dir>/scripts/extract_config.py config.pkl
# Note: zdim, original_Apix values

# 1. Run analysis (generates both analyze.39/ and defanalyze.39/)
dsdsh analyze . 39 10 30

# 2. Extract k-means center 17 as template (using Python)
#    Use the zdim from extract_config.py output
python3 << 'PYEOF'
import pickle
import numpy as np
centers = pickle.load(open('analyze.39/kmeans30/centers.pkl', 'rb'))
center_17 = centers[17]
zdim = len(center_17)
with open('template_z17.txt', 'w') as f:
    f.write(' '.join([f'{v:.6f}' for v in center_17]) + '\n')
    f.write(' '.join(['0.0'] * zdim) + '\n')
PYEOF

# 3. Generate deformation volumes for each PC
for pc in pc1 pc2 pc3 pc4; do
    mkdir -p defanalyze.39_volumes/$pc
    dsd eval_vol --load weights.39.pkl -c config.pkl \
        -o defanalyze.39_volumes/$pc \
        --deform --masks ../mask_params.pkl \
        --template-z template_z17.txt --template-z-ind 0 \
        --zfile defanalyze.39/$pc/z_pc.txt \
        --Apix <original_apix> --prefix reference
done
```

## Key Differences: analyze vs defanalyze Outputs

| Aspect | `analyze.<epoch>/` | `defanalyze.<epoch>/` |
|--------|-------------------|----------------------|
| Purpose | Full composition latent space | Deformation parameter latent space |
| z-dim | Model zdim (from config) | conformational zdim |
| Use with | Standard eval_vol | eval_vol --deform |
| Template needed | No | Yes (from analyze k-means) |

**Note:** Both are generated by a single `dsdsh analyze` command when the model has deformation parameters.

**Dimensionality Reference:**
```bash
# Check your model's zdim
python <skill-dir>/scripts/extract_config.py config.pkl
# Look for: zdim = config['model_args']['zdim']

```

## See Also

- `references/commands.md` - Detailed command reference with all options
- `<skill-dir>/scripts/extract_config.py` - Extract config parameters
- `references/deformation_analysis.md` - Advanced deformation workflows (if available)
