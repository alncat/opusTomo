#!/bin/bash
#SBATCH --job-name=warp_aretomo_neg      # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --time=12:00:00                  # Time limit
#SBATCH --output=logs/warp_aretomo_neg_%j.out # Standard output log
#SBATCH --error=logs/warp_aretomo_neg_%j.err  # Standard error log

# =============================================================================
# WARP AreTomo2 Alignment Script (with Angle Negation)
# 
# Workflow:
# 1. Negate rawtlt angles (WARP convention -> AreTomo convention)
# 2. Run AreTomo2 with negated angles
# 3. Negate AreTomo output angles back (AreTomo convention -> WARP convention)
# =============================================================================

echo "============================================================"
echo "WARP AreTomo2 Alignment (Negation Method)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf defaults)
GPU_ID=0                                         # GPU device ID
MAX_TS=0                                         # 0 = no limit; small value (e.g. 2) to validate on a subset

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo ""

echo "Setting up environment..."

[ -n "$MODULE_CUDA" ]    && module load "$MODULE_CUDA"
[ -n "$MODULE_COMPILER" ] && module load "$MODULE_COMPILER"

# Verify AreTomo2
if [ ! -x "$ARETOMO_EXE" ]; then
    echo "ERROR: AreTomo2 not found at $ARETOMO_EXE"
    exit 1
fi
echo "AreTomo2: $ARETOMO_EXE"

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"

# -----------------------------------------------------------------------------
# Calculate AreTomo2 Parameters
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Alignment Configuration"
echo "============================================================"

# AreTomo2 parameters (in binned pixels — AreTomo runs on the binned stack)
VOL_Z=$((TOMO_DIM_Z / BINNING_FACTOR))
ALIGN_Z=$((VOL_Z * 40 / 100))

# Ensure minimum values
if [ $VOL_Z -lt 200 ]; then
    VOL_Z=200
fi
if [ $ALIGN_Z -lt 80 ]; then
    ALIGN_Z=80
fi

echo "Unbinned Z: ${TOMO_DIM_Z}    Binned Z (${BINNING_FACTOR}x): ${VOL_Z}"
echo ""
echo "AreTomo2 parameters:"
echo "  - VolZ: ${VOL_Z} (reconstruction thickness)"
echo "  - AlignZ: ${ALIGN_Z} (alignment thickness)"
echo "  - OutBin: 1 (no additional binning)"
echo "  - GPU: ${GPU_ID}"

# Check tiltstack directory
if [ ! -d "$TILTSTACK_DIR" ]; then
    echo "ERROR: Tilt stack directory not found: $TILTSTACK_DIR"
    exit 1
fi

# Count tilt series
TS_COUNT=$(find "$TILTSTACK_DIR" -name "*.st" 2>/dev/null | wc -l)
echo ""
echo "Found ${TS_COUNT} tilt series to process"

if [ $TS_COUNT -eq 0 ]; then
    echo "ERROR: No .st files found in $TILTSTACK_DIR"
    exit 1
fi

# -----------------------------------------------------------------------------
# Process Each Tilt Series
# -----------------------------------------------------------------------------
PROCESSED=0
FAILED=0

mapfile -t stack_files < <(ls "$TILTSTACK_DIR"/*/*.st 2>/dev/null | sort)
if [ "${MAX_TS:-0}" -gt 0 ] && [ "${#stack_files[@]}" -gt "$MAX_TS" ]; then
    echo "MAX_TS=$MAX_TS — limiting to first $MAX_TS tilt series (subset test)"
    stack_files=("${stack_files[@]:0:$MAX_TS}")
fi

for stack_file in "${stack_files[@]}"; do
    if [ ! -f "$stack_file" ]; then
        continue
    fi
    
    ts_dir=$(dirname "$stack_file")
    ts_name=$(basename "$stack_file" .st)
    
    echo ""
    echo "----------------------------------------"
    echo "Processing: ${ts_name}"
    echo "Time: $(date +%H:%M:%S)"
    echo "----------------------------------------"
    
    # Check for rawtlt file
    rawtlt_file="${ts_dir}/${ts_name}.rawtlt"
    if [ ! -f "$rawtlt_file" ]; then
        echo "  ERROR: Raw tilt file not found: ${ts_name}.rawtlt"
        ((FAILED++))
        continue
    fi
    
    # Count tilts
    num_tilts=$(wc -l < "$rawtlt_file")
    echo "  Number of tilts: $num_tilts"
    
    # Change to tilt series directory
    cd "$ts_dir" || continue
    
    # =============================================================================
    # Step 1: Negate Raw Tilt Angles (WARP -> AreTomo convention)
    # =============================================================================
    echo "  Step 1: Negating raw tilt angles..."
    
    # Negate for AreTomo: WARP writes +max → -min (e.g. +44..-42); AreTomo expects -max → +min (e.g. -44..+42)
    awk '{print -$1}' "${ts_name}.rawtlt" > "${ts_name}_neg.rawtlt"
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Failed to negate raw tilt angles"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    echo "  Created: ${ts_name}_neg.rawtlt (negated for AreTomo)"
    
    # =============================================================================
    # Step 2: Run AreTomo2
    # =============================================================================
    echo "  Step 2: Running AreTomo2..."
    
    $ARETOMO_EXE \
    -InMrc "${ts_name}.st" \
    -OutMrc "${ts_name}_ali.mrc" \
    -OutBin 1 \
    -AngFile "${ts_name}_neg.rawtlt" \
    -OutImod 2 \
    -VolZ "$VOL_Z" \
    -Gpu "$GPU_ID" \
    -AlignZ "$ALIGN_Z" \
    -FlipVol 1 \
    -FlipInt 1 \
    -DarkTol 0.1 \
    -Wbp 1
    
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "  ERROR: AreTomo2 failed (exit code: ${exit_code})"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    echo "  AreTomo2 completed successfully"
    
    # =============================================================================
    # Step 3: Negate AreTomo Output and Prepare for WARP
    # =============================================================================
    echo "  Step 3: Preparing alignment files for WARP..."
    
    imod_dir="${ts_name}_Imod"
    
    if [ ! -d "$imod_dir" ]; then
        echo "  WARNING: IMOD directory not found: ${imod_dir}"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    # Check for alignment files
    tlt_source="${imod_dir}/${ts_name}_st.tlt"
    xf_source="${imod_dir}/${ts_name}_st.xf"
    
    if [ ! -f "$tlt_source" ]; then
        echo "  WARNING: Tilt file not found: ${tlt_source}"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    if [ ! -f "$xf_source" ]; then
        echo "  WARNING: Transform file not found: ${xf_source}"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    # Negate AreTomo output angles back to WARP convention
    # Just negate the values (AreTomo positive -> WARP negative)
    # Order is preserved (both are ascending in their respective sign conventions)
    awk '{print -$1}' "$tlt_source" > "${ts_name}.tlt"
    
    if [ $? -eq 0 ]; then
        echo "  Created: ${ts_name}.tlt (negated for WARP)"
    else
        echo "  ERROR: Failed to create ${ts_name}.tlt"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    # Copy transform file
    cp "$xf_source" "${ts_name}.xf"
    
    if [ $? -eq 0 ]; then
        echo "  Created: ${ts_name}.xf (order preserved)"
    else
        echo "  ERROR: Failed to create ${ts_name}.xf"
        ((FAILED++))
        cd "$WORK_DIR"
        continue
    fi
    
    ((PROCESSED++))
    cd "$WORK_DIR"
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "AreTomo2 Alignment Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Successfully processed: ${PROCESSED} tilt series"
echo "  - Failed: ${FAILED} tilt series"
echo ""
echo "Output files:"
echo "  - Aligned stacks: ${TILTSTACK_DIR}/*/*_ali.mrc"
echo "  - Alignment files (.xf, .tlt): ${TILTSTACK_DIR}/*/"
echo ""
echo "Next step:"
echo "  Import alignments into WARP:"
echo "  WarpTools ts_import_alignments \\"
echo "      --settings warp_tiltseries.settings \\"
echo "      --alignments ${TILTSTACK_DIR} \\"
echo "      --alignment_angpix <ALIGN_ANGPIX>   # binned pixel size used by AreTomo"
echo "============================================================"
