#!/bin/bash
#SBATCH --job-name=gen_template          # Job name
#SBATCH --partition=normal               # CPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit
#SBATCH --output=logs/gen_template_%j.out     # Standard output log
#SBATCH --error=logs/gen_template_%j.err      # Standard error log

# =============================================================================
# Generate PyTom Template from a User-Provided MRC Map
#
# The output template should match the pixel size used for template matching:
# ALIGN_ANGPIX = ANGPIX * BINNING_FACTOR.
# =============================================================================

echo "============================================================"
echo "Generate PyTom Template from MRC"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf defaults)
# INPUT_MRC="/absolute/path/to/input_map.mrc"
# MAP_ANGPIX="1.5"

# Derived from pipeline.conf — output paths
OUTPUT_DIR="$TEMPLATES_DIR"
OUTPUT_NAME="${TM_LABEL}_tm.mrc"
OUTPUT_TEMPLATE="${OUTPUT_DIR}/${OUTPUT_NAME}"

# Optional create_template.py controls (TEMPLATE_INVERT, TEMPLATE_MIRROR from species.conf)
TEMPLATE_INVERT="${TEMPLATE_INVERT:-0}"
TEMPLATE_MIRROR="${TEMPLATE_MIRROR:-0}"
CORES="${SLURM_CPUS_PER_TASK:-1}"
GPU_ID=""                                # Empty = CPU/default; set e.g. 0 to pass --gpuID 0

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
if ! conda activate "$PYTOM_ENV"; then
    echo "ERROR: Failed to activate PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi
ulimit -v unlimited

cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR"

echo "Working directory: $(pwd)"
echo "Output directory: $OUTPUT_DIR"
echo ""

if ! command -v create_template.py >/dev/null 2>&1; then
    echo "ERROR: create_template.py not found after activating PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi

if [ ! -f "$INPUT_MRC" ]; then
    echo "ERROR: Input MRC not found: $INPUT_MRC"
    exit 1
fi

for required_value in "$ANGPIX" "$BINNING_FACTOR" "$MAP_ANGPIX" "$TM_BOX_SIZE"; do
    if [[ "$required_value" == \<*\> ]] || [ -z "$required_value" ]; then
        echo "ERROR: ANGPIX, BINNING_FACTOR, MAP_ANGPIX, and TM_BOX_SIZE must be set before submission."
        exit 1
    fi
done

ALIGN_ANGPIX=$(awk "BEGIN {printf \"%.6f\", $ANGPIX * $BINNING_FACTOR}")
OUTPUT_TEMPLATE="${OUTPUT_DIR}/${OUTPUT_NAME}"

echo "Configuration:"
echo "  Input map:        $INPUT_MRC"
echo "  Input map apix:   $MAP_ANGPIX A"
echo "  Dataset apix:     $ANGPIX A"
echo "  Binning factor:   $BINNING_FACTOR"
echo "  TM/template apix: $ALIGN_ANGPIX A"
echo "  TM box size:      $TM_BOX_SIZE px"
echo "  Output template:  $OUTPUT_TEMPLATE"
echo ""

# -----------------------------------------------------------------------------
# Generate Template
# -----------------------------------------------------------------------------

CREATE_ARGS=(
-f "$INPUT_MRC"
-d "$OUTPUT_DIR"
-o "$OUTPUT_NAME"
-s "$ANGPIX"
--map-spacing "$MAP_ANGPIX"
-b "$BINNING_FACTOR"
-x "$TM_BOX_SIZE"
--cores "$CORES"
)

if [ -n "$LPF_RESOLUTION" ]; then
    CREATE_ARGS+=(--lpf-resolution "$LPF_RESOLUTION")
fi
if [ "$TEMPLATE_INVERT" = "1" ]; then
    CREATE_ARGS+=(--invert)
fi
if [ "$TEMPLATE_MIRROR" = "1" ]; then
    CREATE_ARGS+=(--mirror)
fi
if [ -n "$GPU_ID" ]; then
    CREATE_ARGS+=(--gpuID "$GPU_ID")
fi

echo "Command:"
printf '  create_template.py'
printf ' %q' "${CREATE_ARGS[@]}"
echo ""
echo ""

if ! create_template.py "${CREATE_ARGS[@]}"; then
    echo "ERROR: create_template.py failed"
    exit 1
fi

if [ ! -f "$OUTPUT_TEMPLATE" ]; then
    echo "ERROR: Expected template was not created: $OUTPUT_TEMPLATE"
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Template Generation Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output template: $OUTPUT_TEMPLATE"
echo ""
echo "Next steps:"
echo "  1. Set TEMPLATE_MRC in gen_sphere_mask.slurm and gen_tm_jobs_aretomo.slurm to:"
echo "     $OUTPUT_TEMPLATE"
echo "  2. Set MASK_MRC to one shared mask path in gen_sphere_mask.slurm and gen_tm_jobs_aretomo.slurm."
echo "  3. Generate a matching sphere mask:"
echo "     sbatch scripts/gen_sphere_mask.slurm"
echo "============================================================"
