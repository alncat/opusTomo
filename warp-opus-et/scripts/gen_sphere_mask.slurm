#!/bin/bash
#SBATCH --job-name=gen_mask              # Job name
#SBATCH --partition=normal               # CPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=00:30:00                  # Time limit (30 minutes)
#SBATCH --output=logs/gen_mask_%j.out         # Standard output log
#SBATCH --error=logs/gen_mask_%j.err          # Standard error log

# =============================================================================
# Generate Sphere Mask for Template Matching
# Creates a spherical mask matching the template dimensions
# =============================================================================

echo "============================================================"
echo "Generate Sphere Mask Started"
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
# MASK_RADIUS, MASK_SIGMA come from species.conf
MASK_SIGMA="${MASK_SIGMA:-1}"            # Edge softness (pixels)

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------
echo ""

echo "Setting up environment..."

source ~/.bashrc
conda activate "$PYTOM_ENV"
ulimit -v unlimited

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"

if ! command -v headerPyTom >/dev/null 2>&1; then
    echo "ERROR: headerPyTom not found after activating PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi
if ! command -v create_mask.py >/dev/null 2>&1; then
    echo "ERROR: create_mask.py not found after activating PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi

# Check template exists
if [[ "$TEMPLATE_MRC" == \<* ]] || [ -z "$TEMPLATE_MRC" ]; then
    echo "ERROR: Set TEMPLATE_MRC to the TM-ready template MRC."
    exit 1
fi

if [ ! -f "$TEMPLATE_MRC" ]; then
    echo "ERROR: Template not found: $TEMPLATE_MRC"
    exit 1
fi

if [ -z "$TM_MASK_MRC" ]; then
    echo "ERROR: TM_MASK_MRC must not be empty."
    exit 1
fi
mask_dir=$(dirname "$TM_MASK_MRC")
if [ -n "$mask_dir" ] && [ "$mask_dir" != "." ]; then
    mkdir -p "$mask_dir"
fi

# Get template dimensions from the MRC header. The template should be cubic.
# This script runs in the PyTom env, so use PyTom's header reader instead of
# assuming OPUS-ET/cryodrgn is installed in the same environment.
header_output=$(headerPyTom "$TEMPLATE_MRC" 2>/dev/null)
size_line=$(echo "$header_output" | grep 'Number of columns')
TEMPLATE_DIMS=$(echo "$size_line" | awk '{print $(NF-2), $(NF-1), $NF}')
TEMPLATE_NX=$(echo "$TEMPLATE_DIMS" | awk '{print $1}')
TEMPLATE_NY=$(echo "$TEMPLATE_DIMS" | awk '{print $2}')
TEMPLATE_NZ=$(echo "$TEMPLATE_DIMS" | awk '{print $3}')

if [ -z "$TEMPLATE_NX" ] || [ -z "$TEMPLATE_NY" ] || [ -z "$TEMPLATE_NZ" ]; then
    echo "ERROR: Could not read template dimensions with headerPyTom"
    echo "$header_output"
    exit 1
fi
if [ "$TEMPLATE_NX" -ne "$TEMPLATE_NY" ] || [ "$TEMPLATE_NX" -ne "$TEMPLATE_NZ" ]; then
    echo "ERROR: Template must be cubic for PyTom create_mask.py; got ${TEMPLATE_NX}x${TEMPLATE_NY}x${TEMPLATE_NZ}"
    exit 1
fi
TEMPLATE_DIM="$TEMPLATE_NX"

echo "Template: $TEMPLATE_MRC"
echo "Dimensions: ${TEMPLATE_NX}x${TEMPLATE_NY}x${TEMPLATE_NZ}"

# Auto-detect ALIGN_ANGPIX from the template MRC header spacing.
# The template is already resampled to ALIGN_ANGPIX by gen_template_from_mrc,
# so its pixel spacing is the ground truth. Only fill in if still a placeholder.
if [[ "$ALIGN_ANGPIX" == \<* ]] || [ -z "$ALIGN_ANGPIX" ]; then
    ALIGN_ANGPIX=$(echo "$header_output" | grep -i 'spacing' | awk '{print $(NF-2)}')
    if [ -z "$ALIGN_ANGPIX" ]; then
        echo "ERROR: Could not auto-detect ALIGN_ANGPIX from template header spacing."
        echo "       Set ALIGN_ANGPIX manually, or re-run gen_template_from_mrc.slurm."
        exit 1
    fi
    echo "Auto-detected ALIGN_ANGPIX=${ALIGN_ANGPIX} A from template header"
fi

# Derive default mask radius from the physical particle radius at the TM pixel size.
if [ -z "$MASK_RADIUS" ]; then
    if [[ "$DIAMETER" == \<* ]] || [ -z "$DIAMETER" ]; then
        echo "ERROR: Set DIAMETER, or set MASK_RADIUS explicitly."
        exit 1
    fi
    MASK_RADIUS=$(awk -v d="$DIAMETER" -v apix="$ALIGN_ANGPIX" 'BEGIN {printf "%d", (d / 2 / apix) + 0.5}')
    echo "Derived MASK_RADIUS=${MASK_RADIUS} px from DIAMETER=${DIAMETER} A and ALIGN_ANGPIX=${ALIGN_ANGPIX} A/px"
fi

if ! [[ "$MASK_RADIUS" =~ ^[0-9]+$ ]] || [ "$MASK_RADIUS" -le 0 ]; then
    echo "ERROR: MASK_RADIUS must be a positive integer pixel radius, got: $MASK_RADIUS"
    exit 1
fi

if ! [[ "$MASK_SIGMA" =~ ^[0-9]+$ ]] || [ "$MASK_SIGMA" -lt 0 ]; then
    echo "ERROR: MASK_SIGMA must be a non-negative integer softness in pixels, got: $MASK_SIGMA"
    exit 1
fi

MAX_RADIUS=$((TEMPLATE_DIM / 2))
MASK_EXTENT=$((MASK_RADIUS + MASK_SIGMA))
if [ "$MASK_EXTENT" -ge "$MAX_RADIUS" ]; then
    echo "ERROR: MASK_RADIUS + MASK_SIGMA = $MASK_EXTENT px must be smaller than half the template box ($MAX_RADIUS px)."
    echo "       Increase TM_BOX_SIZE when generating the template, reduce MASK_RADIUS, or reduce MASK_SIGMA."
    exit 1
fi

# -----------------------------------------------------------------------------
# Generate Mask
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Generating Sphere Mask"
echo "============================================================"
echo "Box size: $TEMPLATE_DIM"
echo "Radius: $MASK_RADIUS"
echo "Sigma: $MASK_SIGMA"
echo "Output: $TM_MASK_MRC"
echo ""

if ! create_mask.py -o "$TM_MASK_MRC" -b "$TEMPLATE_DIM" -r "$MASK_RADIUS" -s "$MASK_SIGMA"; then
    echo "ERROR: Failed to create mask"
    exit 1
fi

if [ ! -f "$TM_MASK_MRC" ]; then
    echo "ERROR: Mask file not created"
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Mask Generation Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output: $TM_MASK_MRC"
echo ""
echo "Next step: set the same TM_MASK_MRC in gen_tm_jobs_aretomo.slurm, then run template matching"
echo "============================================================"
