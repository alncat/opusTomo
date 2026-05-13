#!/bin/bash
#SBATCH --job-name=train_opuset          # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=16               # CPU cores
#SBATCH --mem=256G                       # Memory
#SBATCH --gres=gpu:4                     # 4 GPUs
#SBATCH --time=120:00:00                 # Time limit (5 days)
#SBATCH --output=logs/train_opuset_%j.out     # Standard output
#SBATCH --error=logs/train_opuset_%j.err      # Standard error

# =============================================================================
# OPUS-ET Training Script
# Training cryo-ET heterogeneity analysis model
# =============================================================================

echo "============================================================"
echo "OPUS-ET Training Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf/species.conf defaults)
POSE_PKL=""                                                 # Pose pickle (auto-generated if empty)

# Multi-body (optional)
MASK_PARAMS=""                                         # Path to mask_params.pkl (optional)
SPLIT_PKL="${OUTPUT_DIR}/deep.pkl"                     # Train/val split pickle (in output directory)

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
conda activate "$OPUSET_ENV"
ulimit -v unlimited

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"
echo ""

# Verify GPU
echo "Checking GPUs..."
nvidia-smi
echo ""

# Verify inputs
echo "Checking inputs..."
if [ ! -f "$EXPORT_STAR" ]; then
    echo "ERROR: STAR file not found: $EXPORT_STAR"
    exit 1
fi
echo "  STAR: $EXPORT_STAR"

if [ ! -d "$DATADIR" ]; then
    echo "ERROR: Subtomogram data directory not found: $DATADIR"
    echo "       Run warp_export_particles.slurm first, or set DATADIR to the WARP subtomogram folder."
    exit 1
fi
echo "  Subtomogram data: $DATADIR"

# Auto-detect data parameters if left blank.
if [ -z "$ANGPIX" ]; then
    echo "Auto-detecting ANGPIX from _rlnDetectorPixelSize in STAR..."
    ANGPIX=$(awk '
    /^_rlnDetectorPixelSize[[:space:]]+#/ {
    split($2, a, "#"); col=a[2]; next
    }
    col && $0 !~ /^_/ && $0 !~ /^#/ && NF >= col {
    print $col; exit
    }
    ' "$EXPORT_STAR")

    if [ -z "$ANGPIX" ]; then
        echo "ERROR: Could not auto-detect ANGPIX from _rlnDetectorPixelSize in $EXPORT_STAR."
        echo "       Set ANGPIX manually, or fix the input STAR file metadata."
        exit 1
    fi
    echo "  From particle STAR: $EXPORT_STAR"
fi

if [ -z "$TILT_RANGE" ] || [ -z "$TILT_STEP" ]; then
    echo "Auto-detecting tilt range/step..."
    # Tilt range: scan ALL .tlt files, take the max absolute angle across all tilt series
    if [ -z "$TILT_RANGE" ]; then
        TILT_RANGE=$(find "$TILTSTACK_DIR" -name "*.tlt" -type f -exec awk '
        NF { a=$1 < 0 ? -$1 : $1; if (a > max) max = a }
        END { if (max != "") print max }
        ' {} \; 2>/dev/null | sort -rn | head -1)
    fi
    # Tilt step: read from the first .tlt file (step should be consistent across the dataset)
    if [ -z "$TILT_STEP" ]; then
        TLT_FILE=$(find "$TILTSTACK_DIR" -name "*.tlt" -type f -print -quit 2>/dev/null)
        if [ -n "$TLT_FILE" ]; then
            echo "  Tilt file (step): $TLT_FILE"
            TILT_STEP=$(awk '
            NF && seen {
            d=$1 - prev
            if (d < 0) d = -d
            if (d > 0 && (min == 0 || d < min)) min = d
            }
            NF { prev=$1; seen=1 }
            END { if (min > 0) printf "%.0f", min }
            ' "$TLT_FILE")
        fi
    fi
    if [ -z "$TILT_RANGE" ] || [ -z "$TILT_STEP" ]; then
        echo "ERROR: Could not parse tilt range/step from .tlt files under $TILTSTACK_DIR."
        echo "       Set TILT_RANGE and TILT_STEP manually in species.conf."
        exit 1
    fi
fi

echo "  ANGPIX: $ANGPIX A"
echo "  Tilt range: +/-$TILT_RANGE degrees"
echo "  Tilt step: $TILT_STEP degrees"

# Ensure training mask exists. If requested, create a default sphere mask from
# one exported subtomogram; dsdsh create_mask uses the input MRC for shape/header.
if [ ! -f "$TRAINING_MASK_MRC" ]; then
    if [ "$AUTO_CREATE_TRAINING_MASK" != "1" ]; then
        echo "ERROR: Training mask not found: $TRAINING_MASK_MRC"
        echo "       Run gen_training_mask.slurm or set TRAINING_MASK_MRC to an existing training mask."
        exit 1
    fi

    echo "Training mask not found; creating default spherical mask."
    SAMPLE_SUBTOMO=""
    SAMPLE_SUBTOMO=$(find "$DATADIR" -name "$SUBTOMO_GLOB" -type f -print -quit 2>/dev/null)

    if [ -z "$SAMPLE_SUBTOMO" ]; then
        echo "ERROR: Could not find an exported subtomogram under: $DATADIR"
        echo "       Run warp_export_particles.slurm first, or set TRAINING_MASK_MRC to an existing mask."
        exit 1
    fi

    MASK_RADIUS=$(awk "BEGIN {printf \"%.3f\", ($SUBTOMO_BOX_SIZE * $MASK_RADIUS_FRACTION) / 2}")
    echo "  Input subtomogram: $SAMPLE_SUBTOMO"
    echo "  Output mask: $TRAINING_MASK_MRC"
    echo "  Subtomogram box size: $SUBTOMO_BOX_SIZE voxels"
    echo "  Mask diameter fraction: $MASK_RADIUS_FRACTION"
    echo "  Sphere radius: $MASK_RADIUS voxels"
    echo "  Soft edge: $MASK_SOFT_EDGE voxels"

    dsdsh create_mask "$SAMPLE_SUBTOMO" \
    -o "$TRAINING_MASK_MRC" \
    --sphere-radius "$MASK_RADIUS" \
    --soft-edge "$MASK_SOFT_EDGE"

    if [ $? -ne 0 ] || [ ! -f "$TRAINING_MASK_MRC" ]; then
        echo "ERROR: Failed to create training mask: $TRAINING_MASK_MRC"
        exit 1
    fi
fi
echo "  Mask: $TRAINING_MASK_MRC"
MASK_ARG="-r $TRAINING_MASK_MRC"

# Generate pose pickle if not provided
if [ -z "$POSE_PKL" ] || [ ! -f "$POSE_PKL" ]; then
    echo ""
    echo "Generating pose pickle from STAR file..."
    POSE_PKL="${EXPORT_STAR%.star}_pose.pkl"
    if ! dsd parse_pose_star "$EXPORT_STAR" \
    -D "$TEMPLATERES" \
    --Apix $ANGPIX \
    -o "$POSE_PKL"; then
    echo "ERROR: Failed to generate pose pickle"
    exit 1
fi
    
if [ ! -f "$POSE_PKL" ]; then
    echo "ERROR: Pose pickle generation failed - file not created: $POSE_PKL"
    exit 1
fi
echo "  Generated: $POSE_PKL"
fi
echo "  Poses: $POSE_PKL"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Optional multi-body parameters
MASKS_ARG=""
if [ -n "$MASK_PARAMS" ] && [ -f "$MASK_PARAMS" ]; then
    MASKS_ARG="--masks $MASK_PARAMS"
    echo "  Multi-body: $MASK_PARAMS"
fi

SPLIT_ARG="--split $SPLIT_PKL"
echo "  Split: $SPLIT_PKL"

echo ""
echo "Training parameters:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  zdim: $ZDIM"
echo "  Output: $OUTPUT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Starting OPUS-ET Training"
echo "============================================================"
echo ""

echo "Command:"
echo "  torchrun --nproc_per_node=4 -m cryodrgn.commands.train_tomo_dist \\"
echo "      $EXPORT_STAR \\"
echo "      --poses $POSE_PKL \\"
echo "      -n $NUM_EPOCHS \\"
echo "      -b $BATCH_SIZE \\"
echo "      --zdim $ZDIM \\"
echo "      --zaffinedim $ZAFFINEDIM \\"
echo "      --lr $LEARNING_RATE \\"
echo "      --num-gpus 4 \\"
echo "      --multigpu \\"
echo "      --beta-control $BETA_CONTROL \\"
echo "      -o $OUTPUT_DIR \\"
echo "      $MASK_ARG \\"
echo "      $SPLIT_ARG \\"
echo "      $MASKS_ARG \\"
echo "      --lamb $LAMB \\"
echo "      --bfactor $BFACTOR \\"
echo "      --valfrac $VAL_FRAC \\"
echo "      --templateres $TEMPLATERES \\"
echo "      --tmp-prefix tmp \\"
echo "      --datadir \$(dirname $DATADIR) \\"
echo "      --angpix $ANGPIX \\"
echo "      --downfrac "${DOWNSAMPLE_FRAC:-1.}" \\"
echo "      --warp \\"
echo "      --tilt-range $TILT_RANGE \\"
echo "      --tilt-step $TILT_STEP \\"
echo "      --ctfalpha 0. \\"
echo "      --ctfbeta 1. \\"
echo "      --estpose"
echo ""

torchrun --nproc_per_node=4 -m cryodrgn.commands.train_tomo_dist \
"$EXPORT_STAR" \
--poses "$POSE_PKL" \
-n "$NUM_EPOCHS" \
-b "$BATCH_SIZE" \
--zdim "$ZDIM" \
--zaffinedim "$ZAFFINEDIM" \
--lr "$LEARNING_RATE" \
--num-gpus 4 \
--multigpu \
--beta-control "$BETA_CONTROL" \
-o "$OUTPUT_DIR" \
$MASK_ARG \
$SPLIT_ARG \
$MASKS_ARG \
--lamb "$LAMB" \
--bfactor "$BFACTOR" \
--valfrac "$VAL_FRAC" \
--templateres "$TEMPLATERES" \
--tmp-prefix tmp \
--datadir "$(dirname "$DATADIR")" \
--angpix "$ANGPIX" \
--downfrac "${DOWNSAMPLE_FRAC:-1.}" \
--warp \
--tilt-range "$TILT_RANGE" \
--tilt-step "$TILT_STEP" \
--ctfalpha 0. \
--ctfbeta 1. \
--estpose

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Complete"
else
    echo "Training Failed (exit code: $EXIT_CODE)"
fi
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Analyze trained model:"
echo "     dsdsh analyze $OUTPUT_DIR $((NUM_EPOCHS-1)) 10 20"
echo ""
echo "  2. Generate volumes:"
echo "     dsd eval_vol --load $OUTPUT_DIR/weights.$((NUM_EPOCHS-1)).pkl -c $OUTPUT_DIR/config.pkl ..."
echo ""
echo "  3. Visualize latent space and classify particles"
echo "============================================================"
