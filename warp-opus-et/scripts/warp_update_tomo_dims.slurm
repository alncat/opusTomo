#!/bin/bash
#SBATCH --job-name=warp_update_dims        # Job name
#SBATCH --partition=normal                # CPU partition
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --mem=16G                         # Memory per node
#SBATCH --time=00:30:00                   # Time limit (30 minutes)
#SBATCH --output=logs/warp_update_dims_%j.out  # Standard output log
#SBATCH --error=logs/warp_update_dims_%j.err   # Standard error log

# =============================================================================
# Update warp_tiltseries.settings with UNBINNED tomo dimensions
# derived from AreTomo's _ali.mrc output.
#
# MUST run AFTER Phase 3 (AreTomo alignment) and BEFORE Phase 4
# (ts_import_alignments). If TOMO_DIMS in settings is wrong, Phase 4
# imports correctly but later reconstruction / particle export use wrong
# coordinates.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf defaults)

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source ~/.bashrc
cd "$WORK_DIR" || exit 1

# -----------------------------------------------------------------------------
# Read binned dims from first available _ali.mrc via headerPyTom (PyTOM env)
# -----------------------------------------------------------------------------
conda activate "$PYTOM_ENV"

if ! command -v headerPyTom >/dev/null 2>&1; then
    echo "ERROR: headerPyTom not found after activating PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi

ali_file=$(find "$TILTSTACK_DIR" -name "*_ali.mrc" -type f -print -quit)
if [ -z "$ali_file" ]; then
    echo "ERROR: no *_ali.mrc found under $TILTSTACK_DIR — run Phase 3 first."
    exit 1
fi

echo "Reading dims from: $ali_file"
header_output=$(headerPyTom "$ali_file")
echo "$header_output"

# headerPyTom prints e.g. "Number of columns, rows, sections .....  1440   1024   174"
# The last three whitespace-separated tokens on that line are NX NY NZ.
size_line=$(echo "$header_output" | grep 'Number of columns')
binned_x=$(echo "$size_line" | awk '{print $(NF-2)}')
binned_y=$(echo "$size_line" | awk '{print $(NF-1)}')
binned_z=$(echo "$size_line" | awk '{print $NF}')

if [ -z "$binned_x" ] || [ -z "$binned_y" ] || [ -z "$binned_z" ]; then
    echo "ERROR: failed to parse headerPyTom output. Check format:"
    echo "$header_output"
    exit 1
fi

unbinned_x=$((binned_x * BINNING_FACTOR))
unbinned_y=$((binned_y * BINNING_FACTOR))
unbinned_z=$((binned_z * BINNING_FACTOR))
NEW_TOMO_DIMS="${unbinned_x}x${unbinned_y}x${unbinned_z}"

echo ""
echo "Binned dims (from AreTomo): ${binned_x} x ${binned_y} x ${binned_z}"
echo "Binning factor: ${BINNING_FACTOR}x"
echo "Unbinned TOMO_DIMS: ${NEW_TOMO_DIMS}"

conda deactivate

# -----------------------------------------------------------------------------
# Re-run create_settings with corrected unbinned dimensions (WARP env)
# -----------------------------------------------------------------------------
conda activate "$WARP_ENV"
ulimit -v unlimited
export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"
export WARP_FORCE_MRC_FLOAT32=1

if [ ! -f "$TILTSERIES_SETTINGS" ]; then
    echo "ERROR: Settings file not found: $TILTSERIES_SETTINGS"
    echo "       Run Phase 2 first, or set TILTSERIES_SETTINGS correctly."
    exit 1
fi

if [ -z "$ANGPIX" ]; then
    ANGPIX=$(sed -n 's/.*Name="PixelSize".*Value="\([0-9.][0-9.]*\)".*/\1/p' "$TILTSERIES_SETTINGS" | head -1)
fi
if [[ "$ANGPIX" == \<* ]] || [ -z "$ANGPIX" ]; then
    echo "ERROR: Could not preserve PixelSize from $TILTSERIES_SETTINGS. Expected a line like:"
    echo '       <Param Name="PixelSize" Value="1.96" />'
    echo "       Set ANGPIX manually if the settings format differs."
    exit 1
fi
echo "Preserving settings PixelSize: ${ANGPIX} A"

# Back up previous settings for safety
cp "$TILTSERIES_SETTINGS" "${TILTSERIES_SETTINGS}.before_update"
echo "Backed up existing settings to ${TILTSERIES_SETTINGS}.before_update"

WarpTools create_settings \
--folder_data tomostar \
--folder_processing warp_tiltseries \
--extension "*.tomostar" \
--angpix "$ANGPIX" \
--exposure "$EXPOSURE" \
--tomo_dimensions "$NEW_TOMO_DIMS" \
--output "$TILTSERIES_SETTINGS"

echo ""
echo "Updated $TILTSERIES_SETTINGS with TOMO_DIMS=${NEW_TOMO_DIMS}"
echo "You may now run Phase 4: sbatch scripts/warp_import_alignments.slurm"
