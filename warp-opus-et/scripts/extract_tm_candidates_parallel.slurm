#!/bin/bash
#SBATCH --job-name=extract_parallel      # Job name
#SBATCH --partition=normal               # CPU partition (no GPU needed)
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks=1                       # Single task (Python handles multiprocessing)
#SBATCH --cpus-per-task=32               # CPUs for parallel extraction
#SBATCH --mem=256G                       # Memory
#SBATCH --time=06:00:00                  # Time limit (6 hours for many tomograms)
#SBATCH --output=logs/extract_parallel_%j.out # Standard output
#SBATCH --error=logs/extract_parallel_%j.err  # Standard error

# =============================================================================
# Extract Candidates from Template Matching Results (Parallel)
# Uses Python multiprocessing for parallel extraction
# =============================================================================

echo "============================================================"
echo "Extract Candidates (Parallel)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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

# Script-specific overrides (uncomment to override species.conf defaults)
# EXTRACT_MASK_RADIUS and MIN_SCORE come from species.conf
EXTRACT_MASK_RADIUS="${EXTRACT_MASK_RADIUS:-14}"  # non-max suppression radius
NUM_JOBS=$SLURM_CPUS_PER_TASK                  # Use all allocated CPUs

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
conda activate "$PYTOM_ENV"
ulimit -v unlimited

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"
echo ""

PYTHON_SCRIPT="$SCRIPT_DIR/scripts/extract_candidates_parallel.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python helper not found: $PYTHON_SCRIPT"
    echo ""
    echo "  This SLURM script and extract_candidates_parallel.py are a coupled"
    echo "  pair — the .slurm file is a thin wrapper around the Python."
    echo "  Make sure both files live in the same directory, or set SCRIPTS_DIR"
    echo "  in this script's CONFIGURATION section to the directory containing"
    echo "  extract_candidates_parallel.py."
    exit 1
fi
chmod +x "$PYTHON_SCRIPT"
echo "Using helper: $PYTHON_SCRIPT"

echo "Configuration:"
echo "  TM directory: $TM_JOBS_DIR"
echo "  Output: $TM_PARTICLES_DIR"
echo "  Candidates per tomogram: $NUM_CANDIDATES"
echo "  Mask radius: $EXTRACT_MASK_RADIUS"
echo "  Parallel jobs: $NUM_JOBS"
echo "  Template: ${TEMPLATE:-<auto>}"
echo ""

# -----------------------------------------------------------------------------
# Run Parallel Extraction
# -----------------------------------------------------------------------------

echo "Starting parallel extraction..."
echo ""

TEMPLATE_ARG=""
if [ -n "$TEMPLATE" ]; then
    TEMPLATE_ARG="--template $TEMPLATE"
fi

python3 "$PYTHON_SCRIPT" \
--tm-dir "$TM_JOBS_DIR" \
--output "$TM_PARTICLES_DIR" \
--num-candidates "$NUM_CANDIDATES" \
--mask-radius "$EXTRACT_MASK_RADIUS" \
--min-score "$MIN_SCORE" \
--jobs "$NUM_JOBS" \
$TEMPLATE_ARG

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Extraction Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "All extractions completed successfully!"
    echo ""
    echo "Output files:"
    ls -lh "${TM_PARTICLES_DIR}"/*_particles.xml 2>/dev/null | head -20
    echo ""
    echo "Next steps:"
    echo "  1. Inspect particle lists in PyTom GUI"
    echo "  2. Convert XML picks to per-tilt-series STAR files:"
    echo "     sbatch scripts/convert_to_star.slurm"
    echo "  3. Convert per-tilt-series STAR files to WARP format:"
    echo "     sbatch scripts/convert_pytom_to_warp.slurm"
else
    echo "WARNING: Some extractions may have failed (exit code: $EXIT_CODE)"
fi

echo "============================================================"
