#!/bin/bash
#SBATCH --job-name=tm_sequential         # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Single node only
#SBATCH --ntasks=4                       # 4 MPI tasks
#SBATCH --ntasks-per-node=4              # All 4 tasks on one node
#SBATCH --cpus-per-task=4                # 4 CPUs per task
#SBATCH --mem=256G                       # Memory
#SBATCH --gres=gpu:4                     # 4 GPUs for all jobs
#SBATCH --time=48:00:00                  # Time limit (48 hours - adjust based on number of jobs)
#SBATCH --output=logs/tm_sequential_%j.out    # Standard output
#SBATCH --error=logs/tm_sequential_%j.err     # Standard error

# =============================================================================
# Run Template Matching Jobs Sequentially on Single Node
# =============================================================================
# This SLURM script runs all template matching jobs one at a time
# on a single node to avoid GPU memory conflicts
# =============================================================================

# Don't use set -e as we handle errors manually in the loop
# set -e

echo "============================================================"
echo "Template Matching - Sequential (SLURM)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
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
# TM_SPLIT_X, TM_SPLIT_Y, TM_SPLIT_Z come from species.conf
TM_SPLIT_X="${TM_SPLIT_X:-2}"
TM_SPLIT_Y="${TM_SPLIT_Y:-2}"
TM_SPLIT_Z="${TM_SPLIT_Z:-1}"
MPI_PROCESSES=4
GPU_IDS="0,1,2,3"
MAX_TS=0                                                # 0 = no limit; small value to validate on a subset

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
conda activate "$PYTOM_ENV"
ulimit -v unlimited

cd "$TM_JOBS_DIR" || { echo "ERROR: Cannot cd to $TM_JOBS_DIR"; exit 1; }

echo "Working directory: $(pwd)"
echo ""

# Verify GPU
echo "Checking GPUs..."
nvidia-smi
echo ""

# Verify PyTom
echo "Checking PyTom..."
if ! which localization.py > /dev/null 2>&1; then
    echo "ERROR: localization.py not found"
    exit 1
fi
echo "localization.py: OK"
echo ""

# -----------------------------------------------------------------------------
# Find Jobs
# -----------------------------------------------------------------------------

echo "Finding template matching jobs..."

mapfile -t JOB_DIRS < <(find "$TM_JOBS_DIR" -name "job.xml" -type f | sort)

NUM_JOBS=${#JOB_DIRS[@]}

if [ $NUM_JOBS -eq 0 ]; then
    echo "ERROR: No job.xml files found in $TM_JOBS_DIR"
    exit 1
fi

echo "Found $NUM_JOBS job(s) to process sequentially"

if [ "${MAX_TS:-0}" -gt 0 ] && [ "$NUM_JOBS" -gt "$MAX_TS" ]; then
    echo "MAX_TS=$MAX_TS — limiting to first $MAX_TS jobs (subset test)"
    JOB_DIRS=("${JOB_DIRS[@]:0:$MAX_TS}")
    NUM_JOBS=${#JOB_DIRS[@]}
fi
echo ""

# -----------------------------------------------------------------------------
# Run Jobs Sequentially
# -----------------------------------------------------------------------------

COMPLETED=0
FAILED=0

for i in "${!JOB_DIRS[@]}"; do
    JOB_XML="${JOB_DIRS[$i]}"
    JOB_DIR=$(dirname "$JOB_XML")
    JOB_NAME=$(basename "$JOB_DIR")
    
    echo "========================================"
    echo "Job $((i+1))/$NUM_JOBS: $JOB_NAME"
    echo "Start time: $(date)"
    echo "========================================"
    echo ""
    
    # Change to job directory
    cd "$JOB_DIR" || { 
    echo "WARNING: Cannot cd to $JOB_DIR, skipping"
    ((FAILED++))
    continue
    }
    
    echo "Directory: $(pwd)"
    echo "Command: mpiexec -n $MPI_PROCESSES --tag-output localization.py -j job.xml -x $TM_SPLIT_X -y $TM_SPLIT_Y -z $TM_SPLIT_Z --gpuID $GPU_IDS"
    echo ""
    
    # Run template matching
    echo "Starting mpiexec..."
    mpiexec -n $MPI_PROCESSES --tag-output \
    localization.py \
    -j "job.xml" \
    -x $TM_SPLIT_X -y $TM_SPLIT_Y -z $TM_SPLIT_Z \
    --gpuID $GPU_IDS
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Job completed successfully"
        ((COMPLETED++))
    else
        echo ""
        echo "✗ Job failed (exit code: $EXIT_CODE)"
        ((FAILED++))
        echo "Continuing to next job..."
    fi
    
    echo ""
    echo "End time: $(date)"
    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo "============================================================"
echo "All Jobs Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Completed: $COMPLETED/$NUM_JOBS"
echo "  - Failed: $FAILED/$NUM_JOBS"
echo ""
echo "Output locations:"
echo "  ${TM_JOBS_DIR}/*/"
echo ""
echo "Next steps:"
echo "  1. Extract candidates from correlation maps using extract_tm_candidates_parallel.slurm"
echo "============================================================"
