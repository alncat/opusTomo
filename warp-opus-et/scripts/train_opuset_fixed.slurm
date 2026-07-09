#!/bin/bash
#SBATCH --job-name=opuset_fixed         # Job name
#SBATCH --partition=normal                 # GPU partition
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks=1                      # Single task; torchrun launches GPU workers
#SBATCH --cpus-per-task=8              # CPU cores
#SBATCH --gres=gpu:4                    # Number of GPUs
#SBATCH --mem=256G                      # Memory
#SBATCH --time=72:00:00                 # Time limit (72 hours)
#SBATCH --output=logs/opuset_fixed_%j.out
#SBATCH --error=logs/opuset_fixed_%j.err

# =============================================================================
# OPUS-ET Training - Fixed Mode (Homogeneous Averaging)
# Uses pre-computed poses with fixed latent code (z=0 for all particles)
# =============================================================================

# Don't use set -e as we handle errors manually
# set -e

echo "============================================================"
echo "OPUS-ET Training - Fixed Mode"
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
FIXED_SUBSET_LABEL="${FIXED_SUBSET_LABEL:-1}"       # From dsdsh convert_star --subset-label <N>
STAR_FILE="${WORK_DIR}/warp_tiltseries/${TM_LABEL}_matching_subset${FIXED_SUBSET_LABEL}.star"
POSE_PKL=""                                      # Pose pickle; auto-generated if empty
OUTPUT_DIR="${WORK_DIR}/opuset/${TM_LABEL}/fixed_subset${FIXED_SUBSET_LABEL}"

# CRITICAL: OPUS-ET computes the CTF at --angpix, so it MUST be the SUBTOMOGRAM pixel
# size (OUTPUT_ANGPIX = the exported subtomos' rlnDetectorPixelSize), NOT the raw
# tilt-series ANGPIX from pipeline.conf. A wrong pixel size mis-scales every CTF zero
# and corrupts the reconstruction. Use OUTPUT_ANGPIX; if unset, fall through to the
# STAR auto-detect below (rlnDetectorPixelSize) — never the raw tilt-series value.
ANGPIX="${OUTPUT_ANGPIX:-}"

echo "Configuration:"
echo "  Star file:     ${STAR_FILE}"
echo "  Pose pkl:      ${POSE_PKL}"
echo "  Mask:          ${TRAINING_MASK_MRC}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Data dir:      ${DATADIR}"
echo "  Zdim:          ${ZDIM}"
echo "  Zaffinedim:    ${ZAFFINEDIM}"
echo "  Epochs:        ${NUM_EPOCHS}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  GPUs:          ${NUM_GPUS}"
echo "  Tilt range:    ${TILT_RANGE}"
echo "  Tilt step:     ${TILT_STEP}"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
conda activate "$OPUSET_ENV"
ulimit -v unlimited

echo "Environment activated"
echo ""

# Change to output directory immediately after environment setup
# This ensures split file, logs, and pose pickle are created in the correct location
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" || { echo "ERROR: Cannot cd to ${OUTPUT_DIR}"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

# -----------------------------------------------------------------------------
# Verify Prerequisites
# -----------------------------------------------------------------------------

echo "Checking prerequisites..."

if [ ! -f "${STAR_FILE}" ]; then
    echo "ERROR: Star file not found: ${STAR_FILE}"
    exit 1
fi

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: Subtomogram data directory not found: ${DATADIR}"
    echo "       Run warp_export_particles.slurm first, or set DATADIR to the WARP subtomogram folder."
    exit 1
fi
echo "Resolved subtomogram data directory: ${DATADIR}"

if [ -z "${ANGPIX}" ]; then
    echo "Auto-detecting ANGPIX from _rlnDetectorPixelSize in STAR..."
    ANGPIX=$(awk '
    /^_rlnDetectorPixelSize[[:space:]]+#/ {
    split($2, a, "#"); col=a[2]; next
    }
    col && $0 !~ /^_/ && $0 !~ /^#/ && NF >= col {
    print $col; exit
    }
    ' "${STAR_FILE}")

    if [ -z "${ANGPIX}" ]; then
        echo "ERROR: Could not auto-detect ANGPIX from _rlnDetectorPixelSize in ${STAR_FILE}."
        echo "       Set ANGPIX manually, or fix the input STAR file metadata."
        exit 1
    fi
    echo "  From particle STAR: ${STAR_FILE}"
fi
echo "Resolved ANGPIX: ${ANGPIX} A"

if [ -z "${TILT_RANGE}" ] || [ -z "${TILT_STEP}" ]; then
    echo "Auto-detecting tilt range/step..."
    # Tilt range: scan ALL .tlt files, take the max absolute angle across all tilt series.
    if [ -z "${TILT_RANGE}" ]; then
        TILT_RANGE=$(find "${TILTSTACK_DIR}" -name "*.tlt" -type f -exec awk '
        NF { a=$1 < 0 ? -$1 : $1; if (a > max) max = a }
        END { if (max != "") print max }
        ' {} \; 2>/dev/null | sort -rn | head -1)
    fi
    # Tilt step: read from the first .tlt file; it should be consistent across the dataset.
    if [ -z "${TILT_STEP}" ]; then
        TLT_FILE=$(find "${TILTSTACK_DIR}" -name "*.tlt" -type f -print -quit 2>/dev/null)
        if [ -n "${TLT_FILE}" ]; then
            echo "  Tilt file (step): ${TLT_FILE}"
            TILT_STEP=$(awk '
            NF && seen {
            d=$1 - prev
            if (d < 0) d = -d
            if (d > 0 && (min == 0 || d < min)) min = d
            }
            NF { prev=$1; seen=1 }
            END { if (min > 0) printf "%.0f", min }
            ' "${TLT_FILE}")
        fi
    fi
    if [ -z "${TILT_RANGE}" ] || [ -z "${TILT_STEP}" ]; then
        echo "ERROR: Could not parse tilt range/step from .tlt files under ${TILTSTACK_DIR}."
        echo "       Set TILT_RANGE and TILT_STEP manually in species.conf."
        exit 1
    fi
fi
echo "Resolved tilt range: +/-${TILT_RANGE} degrees"
echo "Resolved tilt step: ${TILT_STEP} degrees"

if [ -z "${POSE_PKL}" ]; then
    POSE_PKL="${OUTPUT_DIR}/$(basename "${STAR_FILE%.star}")_pose.pkl"
fi

if [ ! -f "${POSE_PKL}" ]; then
    echo "Pose pickle not found: ${POSE_PKL}"
    echo ""
    echo "Generating pose pickle automatically..."
    echo "Command: dsd parse_pose_star ${STAR_FILE} -D ${SUBTOMO_BOX_SIZE} --Apix ${ANGPIX} -o ${POSE_PKL}"
    echo ""

    # -D is the SUBTOMOGRAM box (= lattice D-1, the box OPUS-ET's PoseTracker scales pose
    # translations by), NOT TEMPLATERES (the decoder output size). Harmless for template-
    # matching input (zero initial translations -> 0/D) but wrong for any non-zero shifts.
    if ! dsd parse_pose_star "${STAR_FILE}" \
    -D "${SUBTOMO_BOX_SIZE}" \
    --Apix "${ANGPIX}" \
    -o "${POSE_PKL}" || [ ! -f "${POSE_PKL}" ]; then
        echo "ERROR: Failed to generate pose pickle: ${POSE_PKL}"
        exit 1
    fi
    
echo "Pose pickle generated successfully: ${POSE_PKL}"
echo ""
fi

if [ -z "${TRAINING_MASK_MRC}" ]; then
    echo "ERROR: TRAINING_MASK_MRC is required for training."
    echo "       Set TRAINING_MASK_MRC to the desired output path for an auto-created mask, or to an existing mask."
    exit 1
fi

if [ ! -f "${TRAINING_MASK_MRC}" ]; then
    if [ "${AUTO_CREATE_TRAINING_MASK}" != "1" ]; then
        echo "ERROR: Mask file not found: ${TRAINING_MASK_MRC}"
        echo "       Run gen_training_mask.slurm or set TRAINING_MASK_MRC to an existing training mask."
        exit 1
    fi

    echo "Training mask not found; creating default spherical mask."
    # Exclude 3D CTF volumes (*_ctf_*, half-Fourier -> non-cubic D×D×D/2+1) and the
    # *_average* map; only a real-space particle subtomogram is a valid mask template.
    # (A CTF volume yields a non-cubic mask -> cryodrgn aborts with "Images must be cubic".)
    SAMPLE_SUBTOMO=$(find "${DATADIR}" -name "${SUBTOMO_GLOB}" -type f \
        ! -name "*_ctf_*" ! -name "*_average*" -print -quit 2>/dev/null)

    if [ -z "${SAMPLE_SUBTOMO}" ]; then
        echo "ERROR: Could not find an exported subtomogram under: ${DATADIR}"
        echo "       Run warp_export_particles.slurm first, or set TRAINING_MASK_MRC to an existing mask."
        exit 1
    fi

    MASK_RADIUS=$(awk "BEGIN {printf \"%.3f\", (${SUBTOMO_BOX_SIZE} * ${MASK_RADIUS_FRACTION}) / 2}")
    echo "  Input subtomogram: ${SAMPLE_SUBTOMO}"
    echo "  Output mask: ${TRAINING_MASK_MRC}"
    echo "  Subtomogram box size: ${SUBTOMO_BOX_SIZE} voxels"
    echo "  Mask diameter fraction: ${MASK_RADIUS_FRACTION}"
    echo "  Sphere radius: ${MASK_RADIUS} voxels"
    echo "  Soft edge: ${MASK_SOFT_EDGE} voxels"

    dsdsh create_mask "${SAMPLE_SUBTOMO}" \
    -o "${TRAINING_MASK_MRC}" \
    --sphere-radius "${MASK_RADIUS}" \
    --soft-edge "${MASK_SOFT_EDGE}"

    if [ $? -ne 0 ] || [ ! -f "${TRAINING_MASK_MRC}" ]; then
        echo "ERROR: Failed to create training mask: ${TRAINING_MASK_MRC}"
        exit 1
    fi
fi
MASK_ARG="-r ${TRAINING_MASK_MRC}"

echo "Prerequisites: OK"
echo ""

# Create logs directory (already in output directory)
mkdir -p logs

# Verify environment
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo ""

# -----------------------------------------------------------------------------
# Create Split File in Output Directory
# -----------------------------------------------------------------------------

# Split file must be in output directory
SPLIT_FILE="${OUTPUT_DIR}/deep_split.pkl"

echo "Split file: ${SPLIT_FILE}"
echo ""

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------

echo "Starting OPUS-ET training in FIXED MODE..."
echo ""
echo "Command:"
echo "  torchrun --nproc_per_node=${NUM_GPUS} -m cryodrgn.commands.train_tomo_dist \\"
echo "      ${STAR_FILE} \\"
echo "      --poses ${POSE_PKL} \\"
echo "      -n ${NUM_EPOCHS} \\"
echo "      -b ${BATCH_SIZE} \\"
echo "      --zdim ${ZDIM} \\"
echo "      --zaffinedim ${ZAFFINEDIM} \\"
echo "      --lr ${LEARNING_RATE} \\"
echo "      --num-gpus ${NUM_GPUS} \\"
echo "      --multigpu \\"
echo "      --beta-control ${BETA_CONTROL} \\"
echo "      -o ${OUTPUT_DIR} \\"
echo "      ${MASK_ARG} \\"
echo "      --split ${SPLIT_FILE} \\"
echo "      --lamb ${LAMB} \\"
echo "      --bfactor ${BFACTOR} \\"
echo "      --valfrac 0.0 \\"
echo "      --templateres ${TEMPLATERES} \\"
echo "      --tmp-prefix tmp \\"
echo "      --datadir $(dirname "${DATADIR}") \\"
echo "      --angpix ${ANGPIX} \\"
echo "      --downfrac 1. \\"
echo "      --warp \\"
echo "      --ctfalpha 0. \\"
echo "      --ctfbeta 1. \\"
echo "      --encode-mode fixed \\"
echo "      --tilt-range ${TILT_RANGE} \\"
echo "      --tilt-step ${TILT_STEP} \\"
echo "      --checkpoint 10"
echo ""

torchrun --nproc_per_node=${NUM_GPUS} -m cryodrgn.commands.train_tomo_dist \
"${STAR_FILE}" \
--poses "${POSE_PKL}" \
-n "${NUM_EPOCHS}" \
-b "${BATCH_SIZE}" \
--zdim "${ZDIM}" \
--zaffinedim "${ZAFFINEDIM}" \
--lr "${LEARNING_RATE}" \
--num-gpus "${NUM_GPUS}" \
--multigpu \
--beta-control "${BETA_CONTROL}" \
-o "${OUTPUT_DIR}" \
${MASK_ARG} \
--split "${SPLIT_FILE}" \
--lamb "${LAMB}" \
--bfactor "${BFACTOR}" \
--valfrac 0.0 \
--templateres "${TEMPLATERES}" \
--tmp-prefix tmp \
--datadir "$(dirname "$DATADIR")" \
--angpix "${ANGPIX}" \
--downfrac 1. \
--warp \
--ctfalpha 0. \
--ctfbeta 1. \
--encode-mode fixed \
--tilt-range "${TILT_RANGE}" \
--tilt-step "${TILT_STEP}" \
--checkpoint 10

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Training Complete"
echo "End time: $(date)"
echo "============================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"/weights.*.pkl 2>/dev/null | tail -5
    echo ""
    echo "Next steps:"
    echo "  1. Check convergence: tail -f ${OUTPUT_DIR}/run.log"
    echo "  2. Analyze final epoch:"
    echo "     dsdsh analyze ${OUTPUT_DIR} $((NUM_EPOCHS-1)) 10 20"
    echo "  3. Generate volumes:"
    echo "     dsd eval_vol --load ${OUTPUT_DIR}/weights.$((NUM_EPOCHS-1)).pkl -c ${OUTPUT_DIR}/config.pkl ..."
else
    echo "Status: FAILED (exit code: ${EXIT_CODE})"
fi

echo "============================================================"

exit $EXIT_CODE
