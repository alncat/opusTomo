#!/bin/bash
#SBATCH --job-name=gen_train_mask        # Job name
#SBATCH --partition=normal               # CPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=00:30:00                  # Time limit
#SBATCH --output=logs/gen_train_mask_%j.out   # Standard output log
#SBATCH --error=logs/gen_train_mask_%j.err    # Standard error log

# =============================================================================
# Generate OPUS-ET Training Mask from a Consensus Model
# Uses dsdsh create_mask so the output mask keeps the input MRC shape/header.
# =============================================================================

echo "============================================================"
echo "Generate Training Mask"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override species.conf defaults)
# Morph params (TRAINING_MASK_*) and MASK_SOFT_EDGE, MASK_RADIUS_FRACTION come from species.conf
TRAINING_MASK_DILATE="${TRAINING_MASK_DILATE:-3}"
TRAINING_MASK_CLOSE="${TRAINING_MASK_CLOSE:-1}"
TRAINING_MASK_OPEN="${TRAINING_MASK_OPEN:-0}"
TRAINING_MASK_ERODE="${TRAINING_MASK_ERODE:-0}"
TRAINING_MASK_FILL_HOLES="${TRAINING_MASK_FILL_HOLES:-1}"
TRAINING_MASK_LARGEST_COMPONENT="${TRAINING_MASK_LARGEST_COMPONENT:-1}"
# Advanced (rarely changed):
MODE="sphere"                                # sphere or density
THRESHOLD=""                                 # absolute density threshold; blank = use percentile
PERCENTILE="99.0"                            # percentile threshold when THRESHOLD is blank
USE_ABS=0                                    # 1 = threshold absolute density
SPHERE_RADIUS=""                             # voxels; blank = MASK_RADIUS_FRACTION × min_dim / 2
SPHERE_CENTER=""                             # optional "cz cy cx"; blank = volume center

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

echo ""

echo "Setting up environment..."

source ~/.bashrc
conda activate "$OPUSET_ENV"
cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"
echo ""

if [ -z "$TRAINING_MASK_MRC" ]; then
    echo "ERROR: TRAINING_MASK_MRC must not be empty."
    exit 1
fi

if [ -f "$TRAINING_MASK_MRC" ]; then
    echo "Mask already exists: $TRAINING_MASK_MRC"
    echo "Remove it to regenerate, or point TRAINING_MASK_MRC to a new file."
    ls -lh "$TRAINING_MASK_MRC"
    exit 0
fi

# Use an exported REAL-SPACE particle subtomogram as the header source so the
# mask matches training dimensions (OUTPUT_ANGPIX × SUBTOMO_BOX_SIZE). The
# subtomo dir also holds per-particle 3D CTF volumes (*_ctf_*.mrc) and an
# *_average*.mrc; CTF volumes are half-Fourier and therefore NON-CUBIC (e.g.
# 104x104x53 = D x D x D/2+1). If the glob picks one as the reference the mask
# ends up non-cubic and cryodrgn aborts at load with "Images must be cubic",
# and the sphere-mode radius (derived from the reference's min dimension) would
# be wrong -- so exclude *_ctf_* and *_average* below.
MASK_REFERENCE=""
MASK_REFERENCE=$(find "$DATADIR" -name "$SUBTOMO_GLOB" -type f \
    ! -name "*_ctf_*" ! -name "*_average*" -print -quit 2>/dev/null)

if [ -z "$MASK_REFERENCE" ]; then
    echo "ERROR: No subtomograms found under DATADIR=$DATADIR (glob: $SUBTOMO_GLOB)"
    echo "       Run Phase 7 (warp_export_particles.slurm) first."
    exit 1
fi
echo "Mask reference (for header): $MASK_REFERENCE"

if ! command -v dsdsh >/dev/null 2>&1; then
    echo "ERROR: dsdsh not found after activating $OPUSET_ENV"
    exit 1
fi

# -----------------------------------------------------------------------------
# Build dsdsh create_mask command
# -----------------------------------------------------------------------------

cmd=(dsdsh create_mask "$MASK_REFERENCE" -o "$TRAINING_MASK_MRC")

if [ "$MODE" = "density" ]; then
    cmd+=(--percentile "$PERCENTILE")
    cmd+=(--open "$TRAINING_MASK_OPEN" --close "$TRAINING_MASK_CLOSE")
    cmd+=(--dilate "$TRAINING_MASK_DILATE" --erode "$TRAINING_MASK_ERODE")
    cmd+=(--soft-edge "$MASK_SOFT_EDGE")

    if [ -n "$THRESHOLD" ]; then
        cmd+=(--threshold "$THRESHOLD")
    fi
    if [ "$USE_ABS" = "1" ]; then
        cmd+=(--use-abs)
    fi
    if [ "$TRAINING_MASK_LARGEST_COMPONENT" = "1" ]; then
        cmd+=(--largest-component)
    fi
    if [ "$TRAINING_MASK_FILL_HOLES" = "1" ]; then
        cmd+=(--fill-holes)
    fi
elif [ "$MODE" = "sphere" ]; then
    if [ -z "$SPHERE_RADIUS" ]; then
            dims=$(python - "$MASK_REFERENCE" <<'PY'
import sys
from cryodrgn import mrc
header = mrc.MRCHeader.parse(sys.argv[1])
print(f"{header.fields['nz']} {header.fields['ny']} {header.fields['nx']}")
PY
        )
        if [ -z "$dims" ]; then
            echo "ERROR: Could not read consensus/reference dimensions: $TEMPLATE_MRC"
            exit 1
        fi
        min_dim=$(echo "$dims" | awk '{m=$1; for (i=2; i<=NF; i++) if ($i<m) m=$i; print m}')
        SPHERE_RADIUS=$(awk -v d="$min_dim" -v f="$MASK_RADIUS_FRACTION" 'BEGIN {printf "%.3f", d * f / 2}')
        echo "Derived SPHERE_RADIUS=${SPHERE_RADIUS} voxels from min dimension ${min_dim} and fraction ${MASK_RADIUS_FRACTION}"
    fi

    cmd+=(--sphere-radius "$SPHERE_RADIUS")
    cmd+=(--soft-edge "$MASK_SOFT_EDGE")
    if [ -n "$SPHERE_CENTER" ]; then
        read -r cz cy cx <<< "$SPHERE_CENTER"
        cmd+=(--sphere-center "$cz" "$cy" "$cx")
    fi
else
    echo "ERROR: MODE must be 'density' or 'sphere', got: $MODE"
    exit 1
fi

# -----------------------------------------------------------------------------
# Generate mask
# -----------------------------------------------------------------------------

echo "Configuration:"
echo "  Reference (header): $MASK_REFERENCE"
echo "  Output mask:     $TRAINING_MASK_MRC"
echo "  Mode:            $MODE"
echo "  Soft edge:       $MASK_SOFT_EDGE voxels"
if [ "$MODE" = "density" ]; then
    echo "  Threshold:       ${THRESHOLD:-percentile ${PERCENTILE}}"
    echo "  Largest comp:    $TRAINING_MASK_LARGEST_COMPONENT"
    echo "  Fill holes:      $TRAINING_MASK_FILL_HOLES"
    echo "  Open/close:      $TRAINING_MASK_OPEN/$TRAINING_MASK_CLOSE"
    echo "  Dilate/erode:    $TRAINING_MASK_DILATE/$TRAINING_MASK_ERODE"
else
    echo "  Sphere radius:   $SPHERE_RADIUS voxels"
    echo "  Sphere center:   ${SPHERE_CENTER:-volume center}"
fi
echo ""

echo "Command:"
printf '  %q' "${cmd[@]}"
echo ""
echo ""

if ! "${cmd[@]}"; then
    echo "ERROR: Failed to create training mask"
    exit 1
fi

if [ ! -f "$TRAINING_MASK_MRC" ]; then
    echo "ERROR: Mask file not created: $TRAINING_MASK_MRC"
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Training Mask Generated"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output: $TRAINING_MASK_MRC"
ls -lh "$TRAINING_MASK_MRC"
echo ""
echo "Use this path in train_opuset.slurm / train_opuset_fixed.slurm:"
echo "  TRAINING_MASK_MRC=\"$(pwd)/$TRAINING_MASK_MRC\""
echo "============================================================"
