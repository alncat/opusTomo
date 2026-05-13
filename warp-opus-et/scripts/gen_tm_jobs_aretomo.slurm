#!/bin/bash
#SBATCH --job-name=gen_tm_xml            # Job name
#SBATCH --partition=normal               # CPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit (1 hour)
#SBATCH --output=logs/gen_tm_xml_%j.out       # Standard output log
#SBATCH --error=logs/gen_tm_xml_%j.err        # Standard error log

# =============================================================================
# Generate PyTom Template Matching Job XMLs
# Uses AreTomo-reconstructed tomograms (_ali.mrc files)
# =============================================================================

echo "============================================================"
echo "Generate Template Matching Job XMLs"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf/species.conf defaults)
DEFAULT_WEDGE1=40                                       # Default wedge angle 1
DEFAULT_WEDGE2=50                                       # Default wedge angle 2
MAX_TS=0                                                # 0 = no limit; small value (e.g. 2) to validate on a subset

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

source ~/.bashrc
if ! conda activate "$PYTOM_ENV"; then
    echo "ERROR: Failed to activate PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi
cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }
echo "Working directory: $(pwd)"

if ! command -v headerPyTom >/dev/null 2>&1; then
    echo "ERROR: headerPyTom not found after activating PYTOM_ENV=$PYTOM_ENV"
    exit 1
fi

# Check template exists
if [ ! -f "$TEMPLATE_MRC" ]; then
    echo "ERROR: Template not found: $TEMPLATE_MRC"
    exit 1
fi

# Check mask exists
if [ ! -f "$TM_MASK_MRC" ]; then
    echo "WARNING: Mask not found: $TM_MASK_MRC"
    echo "Please run: sbatch gen_sphere_mask.slurm"
    exit 1
fi

# Create base output directory
mkdir -p "$TM_JOBS_DIR"

echo ""
echo "Configuration:"
echo "  Template: $TEMPLATE_MRC"
echo "  Mask: $TM_MASK_MRC"
echo "  Tomostar dir: $TOMOSTAR_DIR"
echo "  Tomogram dir: $TILTSTACK_DIR"
echo "  Angle list: $ANGLE_LIST"
echo "  Output dir: $TM_JOBS_DIR"
if [ -z "$SEARCH_REGION" ]; then
    echo "  Search region: default full X/Y with Z crop starting at z_start=${DEFAULT_Z_START}"
    echo "  Reminder: DEFAULT_Z_START=${DEFAULT_Z_START} is an assumption; set SEARCH_REGION if your useful density starts elsewhere."
else
    echo "  Search region: $SEARCH_REGION"
fi
echo ""

# Check tomostar directory
if [ ! -d "$TOMOSTAR_DIR" ]; then
    echo "ERROR: Tomostar directory not found: $TOMOSTAR_DIR"
    exit 1
fi

# -----------------------------------------------------------------------------
# Process Each Tomogram (using tomostar as canonical source)
# -----------------------------------------------------------------------------

PROCESSED=0
FAILED=0

# Build sorted list of tomostar files — canonical tilt series list
mapfile -t tomostar_files < <(find "$TOMOSTAR_DIR" -name "*.tomostar" -type f | sort)

if [ ${#tomostar_files[@]} -eq 0 ]; then
    echo "ERROR: No .tomostar files found in $TOMOSTAR_DIR"
    exit 1
fi

echo "Found ${#tomostar_files[@]} tilt series from tomostar list"

if [ "${MAX_TS:-0}" -gt 0 ] && [ "${#tomostar_files[@]}" -gt "$MAX_TS" ]; then
    echo "MAX_TS=$MAX_TS — limiting to first $MAX_TS tilt series (subset test)"
    tomostar_files=("${tomostar_files[@]:0:$MAX_TS}")
fi
echo ""

for tomostar_file in "${tomostar_files[@]}"; do
    ts_name=$(basename "$tomostar_file" .tomostar)

    # Find the AreTomo-reconstructed tomogram using find (safe wildcard matching)
    tomo_file=$(find "$TILTSTACK_DIR/$ts_name" -name "*_ali.mrc" -type f -print -quit 2>/dev/null)

    if [ -z "$tomo_file" ] || [ ! -f "$tomo_file" ]; then
        echo "  Skipping $ts_name — no *_ali.mrc found in $TILTSTACK_DIR/$ts_name"
        ((FAILED++))
        continue
    fi

    tm_dir="${TM_JOBS_DIR}/${ts_name}"

    echo "----------------------------------------"
    echo "Processing: $ts_name"
    echo "Tomogram: $tomo_file"
    
    # Create output directory
    mkdir -p "$tm_dir"
    
    # Get tomogram dimensions from the MRC header. This script runs in the
    # PyTom env, so use PyTom's header reader instead of assuming OPUS-ET is
    # installed in the same environment.
    header_output=$(headerPyTom "$tomo_file" 2>/dev/null)
    size_line=$(echo "$header_output" | grep 'Number of columns')
    tomo_dims=$(echo "$size_line" | awk '{print $(NF-2), $(NF-1), $NF}')
    nx=$(echo "$tomo_dims" | awk '{print $1}')
    ny=$(echo "$tomo_dims" | awk '{print $2}')
    nz=$(echo "$tomo_dims" | awk '{print $3}')
    
    if [ -z "$nx" ] || [ -z "$ny" ] || [ -z "$nz" ]; then
        echo "  ERROR: Could not read tomogram dimensions with headerPyTom"
        echo "$header_output"
        ((FAILED++))
        continue
    fi
    
    echo "  Dimensions: ${nx}x${ny}x${nz}"
    
    # Get tilt angles for wedge calculation
    tlt_file="${TILTSTACK_DIR}/${ts_name}/${ts_name}.tlt"
    if [ -f "$tlt_file" ]; then
        # Read first and last angles
        min_angle=$(head -1 "$tlt_file" | awk '{print $1}')
        max_angle=$(tail -1 "$tlt_file" | awk '{print $1}')
        
        # Calculate wedge angles (complement to tilt angles)
        angle1=$(echo "$max_angle" | awk '{printf "%.0f", 90 - ($1 < 0 ? -$1 : $1)}')
        angle2=$(echo "$min_angle" | awk '{printf "%.0f", 90 - ($1 < 0 ? -$1 : $1)}')
        
        # Ensure Angle1 <= Angle2
        if [ "$angle1" -gt "$angle2" ]; then
            tmp=$angle1
            angle1=$angle2
            angle2=$tmp
        fi
    else
        # Use default wedge angles
        angle1=$DEFAULT_WEDGE1
        angle2=$DEFAULT_WEDGE2
        echo "  WARNING: Tilt file not found, using default wedge angles"
    fi
    
    echo "  Wedge angles: [$angle1, $angle2]"
    
    # Determine subregion (PyTom format: " x_start,y_start,z_start,x_dim,y_dim,z_dim ")
    # First 3 numbers = starting coordinates (x,y,z)
    # Second 3 numbers = dimensions (nx,ny,nz)
    if [ -n "$SEARCH_REGION" ]; then
        subregion=" $SEARCH_REGION "
    else
        # Default: crop Z dimension (start at DEFAULT_Z_START, height nz - 2*z_start)
        # Keeps full X and Y
        z_start=$DEFAULT_Z_START
        max_z_dim=$((nz - z_start))
        if [ "$max_z_dim" -le 0 ]; then
            echo "  ERROR: DEFAULT_Z_START=$z_start is outside tomogram Z dimension nz=$nz"
            ((FAILED++))
            continue
        fi
        z_dim=$((nz - 2 * z_start))
        if [ $z_dim -lt 100 ]; then
            z_dim=100  # Preferred minimum Z dimension when it fits
        fi
        if [ "$z_dim" -gt "$max_z_dim" ]; then
            z_dim=$max_z_dim
        fi
        subregion=" 0,0,$z_start,$nx,$ny,$z_dim "
        echo "  Using default search subregion with z_start=$z_start: $subregion"
    fi
    
    # Create job XML with full paths
    job_xml="${tm_dir}/job.xml"
    
    cat > "$job_xml" << EOF
<JobDescription Destination="${tm_dir}">
<Volume Filename="${tomo_file}" Subregion="${subregion}"/>
<Reference Weighting="" File="${TEMPLATE_MRC}"/>
<Mask Filename="${TM_MASK_MRC}" Binning="1" isSphere="True"/>
<WedgeInfo Angle1="${angle1}" Angle2="${angle2}" CutoffRadius="0.0" TiltAxis="custom">
<Rotation Z1="0.0" Z2="0.0" X="0.0"/>
</WedgeInfo>
<Angles Type="FromEMFile" File="${ANGLE_LIST}"/>
<Score Type="FLCFScore" Value="-100000000">
<DistanceFunction Deviation="0.0" Mean="0.0" Filename=""/>
</Score>
</JobDescription>
EOF
    
    if [ $? -eq 0 ]; then
        echo "  Created: $job_xml"
        ((PROCESSED++))
    else
        echo "  ERROR: Failed to create job XML"
        ((FAILED++))
    fi
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Job XML Generation Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Successfully created: $PROCESSED job XMLs"
echo "  - Failed: $FAILED"
echo ""
echo "Output directory: $TM_JOBS_DIR"
echo ""
echo "Use run_tm_sequential.slurm to run all jobs sequentially:"
echo "============================================================"
ulimit -v unlimited
