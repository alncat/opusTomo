#!/bin/bash
#SBATCH --job-name=warp_m_export          # Job name
#SBATCH --partition=gpu                      # GPU partition
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --gres=gpu:4                      # Number of GPUs
#SBATCH --mem=128G                        # Memory per node
#SBATCH --time=12:00:00                   # Time limit (12 hours)
#SBATCH --output=logs/warp_m_export_%j.out     # Standard output log
#SBATCH --error=logs/warp_m_export_%j.err      # Standard error log

# =============================================================================
# Export refined particles for exactly one species (after MCore has converged).
#
# Two-step pipeline:
#   1. Convert MCore's WARP-flavored <species>_particles.star to RELION-style
#      <species>_relion.star via `dsdsh convert_warp`. WarpTools
#      ts_export_particles consumes the RELION dialect, not the WARP one.
#   2. Run ts_export_particles on the converted STAR to write subtomograms.
# =============================================================================

# Do not use set -e here; conversion/export steps report explicit failures.

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf/species.conf defaults)
M_DIR="m"
INPUT_PATTERN="*_relion.star"            # produced by the convert_warp step below
OUTPUT_STAR=""                           # blank = warp_tiltseries/<species>_matching_refined.star

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source ~/.bashrc
ulimit -v unlimited

cd "$WORK_DIR" || exit 1

if [ -d "${M_DIR}/species/${SPECIES_BASE}" ]; then
    SPECIES_DIRS=("${M_DIR}/species/${SPECIES_BASE}")
else
    mapfile -t SPECIES_DIRS < <(find "${M_DIR}/species" -maxdepth 1 -type d -name "${SPECIES_BASE}_*" 2>/dev/null | sort)
fi
if [ ${#SPECIES_DIRS[@]} -eq 0 ]; then
    echo "ERROR: species '$SPECIES_BASE' not found under ${M_DIR}/species/"
    exit 1
fi
if [ ${#SPECIES_DIRS[@]} -gt 1 ]; then
    echo "ERROR: SPECIES_BASE='$SPECIES_BASE' matched multiple species directories:"
    printf '  %s\n' "${SPECIES_DIRS[@]}"
    echo "       Export one species at a time. Set SPECIES_BASE to the full hashed directory name,"
    echo "       e.g. SPECIES_BASE='ribosome_1199b7f2'."
    exit 1
fi

SPECIES_DIR="${SPECIES_DIRS[0]}"
SPECIES_NAME=$(basename "$SPECIES_DIR")
if [ -z "$OUTPUT_STAR" ]; then
    OUTPUT_STAR="warp_tiltseries/${SPECIES_NAME}_matching_refined.star"
fi

# -----------------------------------------------------------------------------
# Step 1: convert each <species>_particles.star (WARP) → _relion.star (RELION)
# -----------------------------------------------------------------------------
mapfile -t PARTICLES_STARS < <(ls "${SPECIES_DIR}"/*_particles.star 2>/dev/null)
if [ ${#PARTICLES_STARS[@]} -eq 0 ]; then
    echo "ERROR: no *_particles.star under $SPECIES_DIR — has refinement run yet?"
    exit 1
fi

# dsdsh lives in a different env; activate it for the conversion only.
if ! conda activate "$OPUSET_ENV"; then
    echo "ERROR: Failed to activate OPUSET_ENV=$OPUSET_ENV"
    exit 1
fi
for ps in "${PARTICLES_STARS[@]}"; do
    out="${ps%_particles.star}_relion.star"
    echo "Converting: $(basename "$ps") -> $(basename "$out")"
    if ! dsdsh convert_warp "$ps" -o "$out"; then
        echo "ERROR: convert_warp failed for $ps"
        exit 1
    fi
done

# Sanity check the converted files exist
if ! ls "${SPECIES_DIR}"/${INPUT_PATTERN} >/dev/null 2>&1; then
    echo "ERROR: convert_warp did not produce $INPUT_PATTERN under $SPECIES_DIR"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_STAR")"

# -----------------------------------------------------------------------------
# Step 2: re-export subtomograms with the converted STAR
# -----------------------------------------------------------------------------
conda deactivate || true
if ! conda activate "$WARP_ENV"; then
    echo "ERROR: Failed to activate WARP_ENV=$WARP_ENV"
    exit 1
fi
export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"
export WARP_FORCE_MRC_FLOAT32=1

echo "Exporting refined particles for one species: $SPECIES_NAME"
echo "  source dir:     $SPECIES_DIR"
echo "  pattern:        $INPUT_PATTERN"
echo "  output star:    $OUTPUT_STAR"
echo "  subtomo box:    $SUBTOMO_BOX_SIZE"
echo "  diameter:       $DIAMETER A"
echo "  output_angpix:  $OUTPUT_ANGPIX A"
echo "  coords_angpix:  $COORDS_ANGPIX A"

"${WARP_DIR}/WarpTools" ts_export_particles \
--settings "$TILTSERIES_SETTINGS" \
--input_directory "$SPECIES_DIR" \
--input_pattern "$INPUT_PATTERN" \
--output_star "$OUTPUT_STAR" \
--output_angpix "$OUTPUT_ANGPIX" \
--box "$SUBTOMO_BOX_SIZE" \
--diameter "$DIAMETER" \
--relative_output_paths \
--3d \
--coords_angpix "$COORDS_ANGPIX" \
--output_ctf_csv \
--dont_correct_ctf_3d

echo ""
echo "Done. Refined export complete for $SPECIES_NAME."
echo "Output STAR: $OUTPUT_STAR"
