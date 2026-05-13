#!/bin/bash
#SBATCH --job-name=pytom2warp          # Job name
#SBATCH --partition=normal             # CPU partition
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=4              # CPU cores
#SBATCH --mem=32G                      # Memory
#SBATCH --time=01:00:00                # Time limit (1 hour)
#SBATCH --output=logs/pytom2warp_%j.out     # Standard output
#SBATCH --error=logs/pytom2warp_%j.err      # Standard error

# =============================================================================
# Convert PyTom STAR Files to WARP Format
# Uses dsdsh convert_pytom to make STAR files compatible with WARP
# =============================================================================

echo "============================================================"
echo "Convert PyTom STAR to WARP Format"
echo "Job ID: $SLURM_JOB_ID"
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

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
if ! conda activate "$OPUSET_ENV"; then
    echo "ERROR: Failed to activate OPUSET_ENV=$OPUSET_ENV"
    exit 1
fi
ulimit -v unlimited

cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }

echo "Working directory: $(pwd)"
echo ""

# Verify dsdsh
echo "Checking dsdsh..."
if ! which dsdsh > /dev/null 2>&1; then
    echo "ERROR: dsdsh not found"
    exit 1
fi
echo "dsdsh: OK"
echo ""

# Create output directory
mkdir -p "$TM_WARP_DIR"

# -----------------------------------------------------------------------------
# Find and Convert STAR Files
# -----------------------------------------------------------------------------

echo "Finding per-tilt-series PyTom STAR files..."

if [ ! -d "$TM_STAR_DIR" ]; then
    echo "ERROR: STAR directory not found: $TM_STAR_DIR"
    echo "       Run convert_to_star.slurm first."
    exit 1
fi

mapfile -t STAR_FILES < <(find "$TM_STAR_DIR" -name "*.star" -type f 2>/dev/null | sort)

NUM_FILES=${#STAR_FILES[@]}

if [ $NUM_FILES -eq 0 ]; then
    echo "ERROR: No STAR files found in $TM_STAR_DIR"
    echo "       Run convert_to_star.slurm first."
    exit 1
fi

echo "Input STAR dir: $TM_STAR_DIR"
echo "Found $NUM_FILES STAR file(s)"
echo "Output directory: $TM_WARP_DIR"
echo ""

# -----------------------------------------------------------------------------
# Convert Each File
# -----------------------------------------------------------------------------

COMPLETED=0
FAILED=0

for star_file in "${STAR_FILES[@]}"; do
    base_name=$(basename "$star_file" .star)

    # Extract tilt series name: handles TS_XXX and Position_X style names.
    # Pattern matches the leading "<letters>_<digits>" prefix before any further "_" separator.
    if [[ "$base_name" =~ ^([A-Za-z]+_[0-9]+) ]]; then
        ts_name="${BASH_REMATCH[1]}"
    else
        echo "WARNING: Could not extract tilt series name from $base_name"
        echo "  Expected format: TS_XXX_* or Position_X_* (leading <word>_<number>)"
        ((FAILED++))
        echo ""
        continue
    fi

    # dsdsh convert_pytom creates output in current directory as <ts_name>_norm.star
    expected_output="${ts_name}_norm.star"
    final_output="${TM_WARP_DIR}/${base_name}_warp.star"
    
    echo "----------------------------------------"
    echo "Converting: $base_name"
    echo "  Input: $star_file"
    echo "  Tilt series: $ts_name"
    echo "  Expected output: $expected_output"
    echo "  Command: dsdsh convert_pytom $star_file $ts_name"
    
    # Remove any existing file to avoid confusion
    if [ -f "$expected_output" ]; then
        rm "$expected_output"
    fi
    
    if ! dsdsh convert_pytom "$star_file" "$ts_name"; then
        echo "  ERROR: convert_pytom failed"
        ((FAILED++))
        echo ""
        continue
    fi

    # dsdsh creates output in current directory as <ts_name>_norm.star
    if [ -f "$expected_output" ]; then
        mv "$expected_output" "$final_output"
        echo "  Success: $final_output"
        ((COMPLETED++))
    else
        echo "  ERROR: Expected output $expected_output not found"
        echo "  Files in current directory:"
        ls -la *.star 2>/dev/null || echo "    (none)"
        ((FAILED++))
    fi
    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Conversion Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Successfully converted: $COMPLETED/$NUM_FILES"
echo "  - Failed: $FAILED/$NUM_FILES"
echo ""
echo "Output STAR files:"
ls -lh "${TM_WARP_DIR}"/*_warp.star 2>/dev/null | head -20
echo ""
echo "Next steps:"
echo "  1. Export subtomograms with WARP:"
echo "     sbatch scripts/warp_export_particles.slurm"
echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: One or more tilt series failed conversion."
    echo "========================================"
    exit 1
fi
echo "============================================================"
