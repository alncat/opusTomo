#!/bin/bash
#SBATCH --job-name=convert_star          # Job name
#SBATCH --partition=normal               # CPU partition
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --mem=64G                        # Memory
#SBATCH --time=02:00:00                  # Time limit (2 hours)
#SBATCH --output=logs/convert_star_%j.out     # Standard output
#SBATCH --error=logs/convert_star_%j.err      # Standard error

# =============================================================================
# Convert PyTom Particle XML to STAR Files
# Converts particle lists from PyTom XML format to RELION/WARP STAR format
# =============================================================================

echo "============================================================"
echo "Convert PyTom XML to STAR Files"
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
if ! conda activate "$PYTOM_ENV"; then
    echo "ERROR: Failed to activate PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi
ulimit -v unlimited

cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }

echo "Working directory: $(pwd)"
echo "Output directory: $TM_STAR_DIR"
echo ""

# Verify convert.py
echo "Checking PyTom convert.py..."
if ! which convert.py > /dev/null 2>&1; then
    echo "ERROR: convert.py not found"
    exit 1
fi
echo "convert.py: OK"
echo ""

# Create output directory
mkdir -p "$TM_STAR_DIR"

# -----------------------------------------------------------------------------
# Find and Convert Particle Lists
# -----------------------------------------------------------------------------

echo "Finding particle XML files..."

mapfile -t XML_FILES < <(find "$TM_PARTICLES_DIR" -name "*_particles.xml" -type f 2>/dev/null | sort)

NUM_FILES=${#XML_FILES[@]}

if [ $NUM_FILES -eq 0 ]; then
    echo "ERROR: No particle XML files found in $TM_PARTICLES_DIR"
    echo "Expected: *_particles.xml"
    echo "Please check that extract_tm_candidates_parallel.slurm completed successfully"
    exit 1
fi

echo "Found $NUM_FILES particle list(s)"
echo ""
echo "Conversion parameters:"
echo "  Pixel size: $ANGPIX A"
echo "  Bin factor: $BINNING_FACTOR"
echo "  Output dir: $TM_STAR_DIR"
echo ""

# -----------------------------------------------------------------------------
# Convert Each File
# -----------------------------------------------------------------------------

COMPLETED=0
FAILED=0

for xml_file in "${XML_FILES[@]}"; do
    # Get base name
    base_name=$(basename "$xml_file" "_particles.xml")
    star_file="${TM_STAR_DIR}/${base_name}.star"
    
    echo "----------------------------------------"
    echo "Converting: $base_name"
    echo "  Input: $xml_file"
    echo "  Output: $star_file"

    rm -f "$star_file" "${base_name}.star"
    
    if ! convert.py \
    -f "$xml_file" \
    --outname "$star_file" \
    --pixelSize "$ANGPIX" \
    --binPyTom "$BINNING_FACTOR" \
    -o star; then
    echo "  ERROR: convert.py failed"
    ((FAILED++))
    echo ""
    continue
fi

    # Some convert.py versions write to the current directory even when --outname
    # includes a directory. Move that output into place if needed.
temp_file="${base_name}.star"
if [ -f "$temp_file" ]; then
    mv "$temp_file" "$star_file"
fi

if [ -f "$star_file" ]; then
    echo "  Success: $star_file"
    ((COMPLETED++))
else
    echo "  ERROR: Output file not found: $star_file"
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
ls -lh "${TM_STAR_DIR}"/*.star 2>/dev/null | head -20
echo ""
echo "Notes:"
echo "  - STAR files are saved to: $TM_STAR_DIR"
echo ""
echo "Next steps:"
echo "  1. Convert to WARP STAR files:"
echo "     sbatch scripts/convert_pytom_to_warp.slurm"
echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: One or more XML files failed conversion."
    echo "========================================"
    exit 1
fi
echo "============================================================"
