#!/bin/bash
#SBATCH --job-name=warp_m_refine          # Job name
#SBATCH --partition=gpu                      # GPU partition
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --gres=gpu:4                      # Number of GPUs
#SBATCH --mem=256G                        # Memory per node
#SBATCH --time=24:00:00                   # Time limit (24 hours)
#SBATCH --output=logs/warp_m_refine_%j.out     # Standard output log
#SBATCH --error=logs/warp_m_refine_%j.err      # Standard error log

# =============================================================================
# MCore refinement pass. Iterative — re-submit this script each time you want
# to refine further. To update the mask between passes, run
# scripts/warp_m_update_mask.slurm in between.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf/species.conf defaults)
M_DIR="m"
REFINE_PARTICLES=1                       # 1 = refine particle poses, 0 = skip
CTF_DEFOCUS=1                            # 1 = refine CTF defocus per tilt, 0 = skip
PERDEVICE_REFINE=1                       # threads per GPU

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source ~/.bashrc
conda activate "$WARP_ENV"
ulimit -v unlimited
export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"
export WARP_FORCE_MRC_FLOAT32=1

cd "$WORK_DIR" || exit 1

POPULATION_FILE="${M_DIR}/${POPULATION_NAME}.population"
if [ ! -f "$POPULATION_FILE" ]; then
    echo "ERROR: population not found: $POPULATION_FILE"
    exit 1
fi

# -----------------------------------------------------------------------------
# Build MCore command
# -----------------------------------------------------------------------------
MCORE_ARGS=(--population "$POPULATION_FILE" --perdevice_refine "$PERDEVICE_REFINE")
[ -n "$REFINE_IMAGEWARP" ]      && MCORE_ARGS+=(--refine_imagewarp "$REFINE_IMAGEWARP")
[ "$REFINE_PARTICLES" = "1" ]   && MCORE_ARGS+=(--refine_particles)
[ "$CTF_DEFOCUS" = "1" ]        && MCORE_ARGS+=(--ctf_defocus)

echo "Running MCore on $POPULATION_FILE"
echo "  args: ${MCORE_ARGS[*]}"

MCore "${MCORE_ARGS[@]}"

echo ""
echo "Done. Refined half maps in m/species/<name>/<species>_half1.mrc and <species>_half2.mrc."
echo "Next options:"
echo "  - update mask:        sbatch scripts/warp_m_update_mask.slurm"
echo "  - another pass:       sbatch scripts/warp_m_refine.slurm"
echo "  - export refined:     sbatch scripts/warp_m_export.slurm"
