#!/bin/bash
# =============================================================================
# Pipeline Pre-flight Validation — Phase-Aware
# =============================================================================
# Usage:
#   bash validate.sh                          # full validation
#   bash validate.sh --phase 1                # only Phase 1
#   bash validate.sh --phase 6 --species species_ribo.conf  # Phase 6, specific species
#   bash validate.sh --phase 7 --dry-run      # print commands, don't check paths
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$SCRIPT_DIR/scripts"
MANIFEST="$SCRIPTS/manifest.yml"
ERRORS=0
WARNINGS=0

PHASE=""
SPECIES_CONF="${SPECIES_CONF:-$SCRIPT_DIR/species.conf}"
DRY_RUN=0

# --- parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --species) SPECIES_CONF="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown: $1"; exit 2 ;;
    esac
done

PIPELINE_CONF="${PIPELINE_CONF:-$SCRIPT_DIR/pipeline.conf}"

# Step counter for banner progress (dry-run has one extra section)
STEP_NUM=0
if [ "$DRY_RUN" -eq 1 ]; then
    TOTAL_STEPS=9
else
    TOTAL_STEPS=8
fi

# --- helpers ---
red()    { echo -e "\033[31m$1\033[0m"; }
green()  { echo -e "\033[32m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }

banner() {
    ((STEP_NUM+=1))
    echo ""
    echo "============================================================"
    echo "[$STEP_NUM/$TOTAL_STEPS] $1"
    echo "============================================================"
}
fail()   { red "  ✗ $1"; ((ERRORS+=1)); return 0; }
pass()   { green "  ✓ $1"; }
warn()   { yellow "  ⚠ $1"; ((WARNINGS+=1)); return 0; }
info()   { echo "  ⓘ $1"; }

is_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

# Progress bar (only animated when stdout is a terminal)
PROGRESS_TTY=0
[ -t 1 ] && PROGRESS_TTY=1

progress_bar() {
    local current="$1" total="$2" msg="${3:-}"
    local width=30 pct=$((current * 100 / total))
    local filled=$((pct * width / 100))
    local empty=$((width - filled))
    local bar
    bar=$(printf '%*s' "$filled" '' | tr ' ' '=')
    bar+=$(printf '%*s' "$empty" '' | tr ' ' '-')
    printf "\r  [%s] %3d%% (%d/%d)" "$bar" "$pct" "$current" "$total"
    [ -n "$msg" ] && printf "  %s" "$msg"
}

progress_clear() {
    printf "\r%*s\r" 80 ''
}

scan_placeholders() {
    local file="$1"
    awk '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*echo[[:space:]]/ { next }
        {
            line = $0
            sub(/[[:space:]]+#.*/, "", line)
            if (line ~ /\/path\/to\// || line ~ /<[A-Za-z][A-Za-z0-9_.\/ -]*>/) {
                print FNR ":" $0
            }
        }
    ' "$file"
}

need_phase() {
    # exit 0 (true) if the given phase is needed given --phase N
    local p="$1"
    [ -z "$PHASE" ] && return 0
    [ "$PHASE" = "$p" ] && return 0
    return 1
}

# Read simple manifest.yml fields without requiring PyYAML. The manifest uses a
# deliberately small YAML subset: phases -> phase -> scalar/list fields.
manifest_field() {
    local phase="${1:-all}" field="$2"
    [ ! -f "$MANIFEST" ] && return 1
    python3 - "$MANIFEST" "$phase" "$field" <<'PY' 2>/dev/null
import re
import sys

path, phase, field = sys.argv[1:4]
phases = {}
order = []
current_phase = None
current_field = None
in_phases = False

def clean_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if (value[0:1], value[-1:]) in [(('"', '"')), (("'", "'"))]:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value

with open(path, encoding="utf-8") as fh:
    for raw in fh:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^phases:\s*$", line):
            in_phases = True
            continue
        if not in_phases:
            continue

        match = re.match(r"^  ([\"']?[A-Za-z0-9.]+[\"']?):\s*$", line)
        if match:
            key = clean_scalar(match.group(1))
            phases[key] = {}
            order.append(key)
            current_phase = key
            current_field = None
            continue

        match = re.match(r"^    ([A-Za-z_]+):\s*(.*)$", line)
        if match and current_phase is not None:
            current_field, value = match.group(1), clean_scalar(match.group(2))
            if value == "":
                phases[current_phase][current_field] = []
            elif value.lower() in {"true", "false"}:
                phases[current_phase][current_field] = value.lower()
                current_field = None
            else:
                phases[current_phase][current_field] = value
                current_field = None
            continue

        match = re.match(r"^      -\s*(.*)$", line)
        if match and current_phase is not None and current_field is not None:
            phases[current_phase].setdefault(current_field, []).append(clean_scalar(match.group(1)))

def emit(value):
    if isinstance(value, list):
        print("\n".join(str(x) for x in value))
    elif value is not None:
        print(value)

if phase in {"", "all"}:
    items = []
    for key in order:
        value = phases.get(key, {}).get(field, [])
        if isinstance(value, list):
            items.extend(value)
        elif value:
            items.append(value)
    if field == "required_vars":
        items = list(dict.fromkeys(items))
    emit(items)
else:
    emit(phases.get(phase, {}).get(field, ""))
PY
}

phase_scripts() {
    local phase="${1:-all}"
    local scripts
    scripts="$(manifest_field "$phase" scripts)"
    if [ -n "$scripts" ]; then
        echo "$scripts"
        return 0
    fi

    case "$phase" in
        1) echo "warp_frameseries_import.slurm" ;;
        2) echo "warp_tiltseries_setup.slurm" ;;
        3) echo "warp_export_stacks.slurm warp_aretomo_align_negate.slurm" ;;
        3a) echo "warp_export_stacks.slurm" ;;
        3b) echo "warp_aretomo_align_negate.slurm" ;;
        3.5) echo "warp_update_tomo_dims.slurm" ;;
        4) echo "warp_import_alignments.slurm" ;;
        5) echo "warp_ts_ctf.slurm warp_ts_reconstruct.slurm" ;;
        5a) echo "warp_ts_ctf.slurm" ;;
        5b) echo "warp_ts_reconstruct.slurm" ;;
        6) echo "gen_template_from_mrc.slurm gen_sphere_mask.slurm gen_tm_jobs_aretomo.slurm run_tm_sequential.slurm extract_tm_candidates_parallel.slurm convert_to_star.slurm convert_pytom_to_warp.slurm" ;;
        6a) echo "gen_template_from_mrc.slurm" ;;
        6b) echo "gen_sphere_mask.slurm" ;;
        6c) echo "gen_tm_jobs_aretomo.slurm" ;;
        6d) echo "run_tm_sequential.slurm" ;;
        6e) echo "extract_tm_candidates_parallel.slurm" ;;
        6f) echo "convert_to_star.slurm" ;;
        6g) echo "convert_pytom_to_warp.slurm" ;;
        7) echo "warp_export_particles.slurm" ;;
        8) echo "gen_training_mask.slurm train_opuset.slurm train_opuset_fixed.slurm" ;;
        8a) echo "gen_training_mask.slurm" ;;
        8b) echo "train_opuset.slurm" ;;
        8c) echo "train_opuset_fixed.slurm" ;;
        M) echo "warp_m_setup.slurm warp_m_create_species.slurm warp_m_refine.slurm warp_m_update_mask.slurm warp_m_export.slurm" ;;
        *) ls "$SCRIPTS"/*.slurm 2>/dev/null | xargs -n1 basename ;;
    esac
}

phase_requires_species() {
    local phase="${1:-all}"
    local value
    value="$(manifest_field "$phase" requires_species)"
    if [ -n "$value" ]; then
        echo "$value"
        return 0
    fi

    case "$phase" in
        ""|all) echo "true" ;;
        6|6a|6b|6c|6d|6e|6f|6g|7|8|8a|8b|8c|M) echo "true" ;;
        *) echo "false" ;;
    esac
}

phase_required_vars() {
    local phase="${1:-all}"
    local vars
    vars="$(manifest_field "$phase" required_vars)"
    if [ -n "$vars" ]; then
        echo "$vars"
        return 0
    fi
    case "$phase" in
        3a) echo "WORK_DIR WARP_DIR CONDA_LIB BINNING_FACTOR" | tr ' ' '\n' ;;
        3b) echo "WORK_DIR ARETOMO_EXE BINNING_FACTOR TOMO_DIM_Z" | tr ' ' '\n' ;;
        5a) echo "WORK_DIR WARP_DIR CONDA_LIB CTF_RANGE_MAX CTF_DEFOCUS_MAX CTF_WINDOW" | tr ' ' '\n' ;;
        5b) echo "WORK_DIR WARP_DIR CONDA_LIB BINNING_FACTOR" | tr ' ' '\n' ;;
        6a) echo "WORK_DIR ANGPIX BINNING_FACTOR TM_LABEL INPUT_MRC MAP_ANGPIX TM_BOX_SIZE" | tr ' ' '\n' ;;
        6b) echo "WORK_DIR TEMPLATE_MRC TM_MASK_MRC DIAMETER MASK_SIGMA" | tr ' ' '\n' ;;
        6c) echo "WORK_DIR TOMOSTAR_DIR TILTSTACK_DIR TEMPLATE_MRC TM_MASK_MRC ANGLE_LIST DEFAULT_Z_START TM_LABEL" | tr ' ' '\n' ;;
        6d) echo "WORK_DIR TM_JOBS_DIR TM_SPLIT_X TM_SPLIT_Y TM_SPLIT_Z" | tr ' ' '\n' ;;
        6e) echo "WORK_DIR TM_JOBS_DIR TM_PARTICLES_DIR NUM_CANDIDATES EXTRACT_MASK_RADIUS TEMPLATE MIN_SCORE" | tr ' ' '\n' ;;
        6f) echo "WORK_DIR TM_PARTICLES_DIR TM_STAR_DIR ANGPIX BINNING_FACTOR" | tr ' ' '\n' ;;
        6g) echo "WORK_DIR TM_STAR_DIR TM_WARP_DIR OPUSET_ENV" | tr ' ' '\n' ;;
    esac
}

phase_required_tools() {
    local phase="${1:-all}"
    local tools
    tools="$(manifest_field "$phase" required_tools)"
    if [ -n "$tools" ]; then
        echo "$tools"
        return 0
    fi
    case "$phase" in
        3a) echo "WarpTools" ;;
        3b) echo "AreTomo2 headerPyTom" | tr ' ' '\n' ;;
        5a|5b) echo "WarpTools" ;;
        6a) echo "create_template.py" ;;
        6b) echo "headerPyTom create_mask.py" | tr ' ' '\n' ;;
        6c) echo "headerPyTom" ;;
        6d) echo "localization.py" ;;
        6f) echo "convert.py" ;;
        6g) echo "dsdsh" ;;
    esac
}

need_species() {
    # exit 0 (true) if species vars are needed for this --phase
    [ -z "$PHASE" ] && return 0
    local result=$(phase_requires_species "$PHASE")
    [ "$result" = "true" ] && return 0
    return 1
}

need_tool() {
    local tool="$1" phase="$2"
    # only require the tool if validating a phase that needs it
    if need_phase "$phase"; then
        if [ -x "$tool" ] || command -v "$tool" >/dev/null 2>&1; then
            pass "$tool found"
        else
            fail "$tool not found (needed from Phase $phase)"
        fi
    else
        pass "$tool (skipped — Phase $phase)"
    fi
}

need_dir() {
    local path="$1" desc="$2" required_phase="$3"
    if need_phase "$required_phase"; then
        if [ -d "$path" ]; then
            pass "$desc"
        else
            fail "$desc missing: $path (needed for Phase $required_phase)"
        fi
    else
        pass "$desc (skipped — Phase $required_phase)"
    fi
}

maybe_dir() {
    local path="$1" desc="$2" phase="$3"
    if [ -d "$path" ]; then
        pass "$desc: $path"
    elif need_phase "$phase"; then
        warn "$desc not yet created: $path (will be created by scripts)"
    else
        pass "$desc (skipped — Phase $phase)"
    fi
}

# =============================================================================
# 1. Load config
# =============================================================================
banner "1. Loading configuration"

if [ ! -f "$PIPELINE_CONF" ]; then
    fail "pipeline.conf not found (copy pipeline.example.conf -> pipeline.conf and edit it)"
else
    pass "pipeline.conf found"
    source "$PIPELINE_CONF"

    # Ensure logs directory exists (all SLURM scripts write to logs/).
    # Only create it after WORK_DIR has been loaded and sanity-checked enough to
    # avoid accidentally writing to /logs when pipeline.conf is missing.
    if [ -n "$WORK_DIR" ] && [[ "$WORK_DIR" != /path/to/* ]]; then
        mkdir -p "$WORK_DIR/logs" 2>/dev/null || warn "Could not create logs directory: $WORK_DIR/logs"
    fi
fi

if need_species; then
    if [ ! -f "$SPECIES_CONF" ]; then
        fail "Species config not found: $SPECIES_CONF (copy species.example.conf -> species_<label>.conf and pass --species)"
    else
        pass "$(basename "$SPECIES_CONF") found"
        source "$SPECIES_CONF"
    fi
else
    pass "Species config (skipped — Phase ${PHASE:-all})"
fi

# =============================================================================
# 2. Placeholder and manifest checks
# =============================================================================
banner "2. Placeholder and manifest checks"

for conf_file in "$PIPELINE_CONF"; do
    if [ ! -f "$conf_file" ]; then
        continue
    fi
    issues=$(scan_placeholders "$conf_file")
    if [ -n "$issues" ]; then
        fail "Unresolved placeholders in $(basename "$conf_file"):"
        echo "$issues" | while read line; do echo "      $line"; done
    else
        pass "No placeholders in $(basename "$conf_file")"
    fi
done

if need_species && [ -f "$SPECIES_CONF" ]; then
    issues=$(scan_placeholders "$SPECIES_CONF")
    if [ -n "$issues" ]; then
        fail "Unresolved placeholders in $(basename "$SPECIES_CONF"):"
        echo "$issues" | while read line; do echo "      $line"; done
    else
        pass "No placeholders in $(basename "$SPECIES_CONF")"
    fi
fi

if [ -f "$MANIFEST" ]; then
    pass "manifest.yml found"
else
    warn "manifest.yml not found; using built-in phase fallback"
fi

while IFS= read -r var_name; do
    [ -z "$var_name" ] && continue
    if [ -z "${!var_name+x}" ]; then
        fail "Required variable $var_name is not defined for Phase ${PHASE:-all}"
    elif [ -z "${!var_name}" ]; then
        fail "Required variable $var_name is empty for Phase ${PHASE:-all}"
    fi
done < <(phase_required_vars "$PHASE")

for s in $(phase_scripts "$PHASE"); do
    script="$SCRIPTS/$s"
    [ ! -f "$script" ] && fail "Missing script: $s" && continue
    issues=$(scan_placeholders "$script")
    if [ -n "$issues" ]; then
        fail "$s: unresolved placeholders"
        echo "$issues" | while read line; do echo "      $line"; done
    fi
done

# =============================================================================
# 3. Environment paths (always needed)
# =============================================================================
if [ "$DRY_RUN" != "1" ]; then
    banner "3. Environment paths"

    [ -d "$WORK_DIR" ] && pass "WORK_DIR: $WORK_DIR" || fail "WORK_DIR not found: $WORK_DIR"
    [ -x "$WARP_DIR/WarpTools" ] && pass "WarpTools: $WARP_DIR/WarpTools" || fail "WarpTools not found"
    [ -f "$CONDA_LIB/libstdc++.so.6" ] && pass "CONDA_LIB" || fail "CONDA_LIB/libstdc++.so.6 not found"
else
    banner "3. Environment paths"
    info "Skipped in --dry-run mode"
fi

# =============================================================================
# 4. Tools — phase-aware
# =============================================================================
if [ "$DRY_RUN" != "1" ]; then
    banner "4. Required tools"

    while IFS= read -r tool_name; do
        [ -z "$tool_name" ] && continue
        case "$tool_name" in
            WarpTools)
                if [ -x "$WARP_DIR/WarpTools" ]; then
                    pass "WarpTools found: $WARP_DIR/WarpTools"
                else
                    fail "WarpTools not found at WARP_DIR/WarpTools"
                fi
                ;;
            AreTomo2)
                if [ -x "$ARETOMO_EXE" ] || command -v "$ARETOMO_EXE" >/dev/null 2>&1; then
                    pass "AreTomo2 found: $ARETOMO_EXE"
                else
                    fail "AreTomo2 not found: $ARETOMO_EXE"
                fi
                ;;
            dsdsh)
                if command -v dsdsh >/dev/null 2>&1 || \
                   (command -v conda >/dev/null 2>&1 && conda run -n "$OPUSET_ENV" which dsdsh >/dev/null 2>&1); then
                    pass "dsdsh found (in PATH or OPUSET_ENV=$OPUSET_ENV)"
                else
                    fail "dsdsh not found (expected in PATH or OPUSET_ENV=$OPUSET_ENV)"
                fi
                ;;
            MTools|MCore)
                if command -v "$tool_name" >/dev/null 2>&1 || [ -x "$WARP_DIR/$tool_name" ]; then
                    pass "$tool_name found"
                else
                    fail "$tool_name not found (expected in PATH or WARP_DIR)"
                fi
                ;;
            create_template.py|create_mask.py|localization.py|convert.py|extractCandidates.py|headerPyTom)
                if command -v "$tool_name" >/dev/null 2>&1 || \
                   (command -v conda >/dev/null 2>&1 && conda run -n "$PYTOM_ENV" which "$tool_name" >/dev/null 2>&1); then
                    pass "$tool_name found (in PATH or PYTOM_ENV=$PYTOM_ENV)"
                else
                    fail "$tool_name not found (expected in PATH or PYTOM_ENV=$PYTOM_ENV)"
                fi
                ;;
            torchrun)
                if command -v torchrun >/dev/null 2>&1 || \
                   (command -v conda >/dev/null 2>&1 && conda run -n "$OPUSET_ENV" which torchrun >/dev/null 2>&1); then
                    pass "torchrun found (in PATH or OPUSET_ENV=$OPUSET_ENV)"
                else
                    fail "torchrun not found (expected in PATH or OPUSET_ENV=$OPUSET_ENV)"
                fi
                ;;
            *)
                if command -v "$tool_name" >/dev/null 2>&1; then
                    pass "$tool_name found"
                else
                    warn "$tool_name not found by generic PATH check"
                fi
                ;;
        esac
    done < <(phase_required_tools "$PHASE")
else
    banner "4. Required tools"
    info "Skipped in --dry-run mode"
fi

# =============================================================================
# 5. Project directories — phase-aware
# =============================================================================
if [ "$DRY_RUN" != "1" ]; then
    banner "5. Project directories"

    need_dir "$WORK_DIR/mdoc"                         "MDOC dir"                  1
    need_dir "$FRAMESERIES_DIR"                       "Frame series dir"          1
    need_dir "$TOMOSTAR_DIR"                          "Tomostar dir"              2
    need_dir "$TILTSTACK_DIR"                         "Tiltstack dir"             3
    need_dir "$WORK_DIR/warp_tiltseries/reconstruction" "Recon dir"               5
    need_dir "$TEMPLATES_DIR"                         "Templates dir"             6

    maybe_dir "$TM_JOBS_DIR"        "TM jobs dir"        6
    maybe_dir "$TM_PARTICLES_DIR"   "TM particles dir"   6
    maybe_dir "$TM_STAR_DIR"        "TM star_files dir"  6
    maybe_dir "$TM_WARP_DIR"        "TM warp_star dir"   6
    maybe_dir "$DATADIR"            "Subtomogram dir"    7
else
    banner "5. Project directories"
    info "Skipped in --dry-run mode"
fi

# =============================================================================
# 6. Derived values
# =============================================================================
banner "6. Derived value sanity"

echo "  ANGPIX:           $ANGPIX Å"
echo "  BINNING_FACTOR:   ${BINNING_FACTOR}x"
echo "  ALIGN_ANGPIX:     $ALIGN_ANGPIX Å"
echo "  TOMO_DIMS:        ${TOMO_DIM_X:-?} × ${TOMO_DIM_Y:-?} × ${TOMO_DIM_Z:-?} (unbinned)"

# If AreTomo output exists, show actual binned dims
ali_sample=$(find "$TILTSTACK_DIR" -name "*_ali.mrc" -type f -print -quit 2>/dev/null)
if [ -n "$ali_sample" ]; then
    ali_info=$(conda run -n "$PYTOM_ENV" headerPyTom "$ali_sample" 2>/dev/null | grep 'Number of columns' || true)
    if [ -n "$ali_info" ]; then
        bnx=$(echo "$ali_info" | awk '{print $(NF-2)}')
        bny=$(echo "$ali_info" | awk '{print $(NF-1)}')
        bnz=$(echo "$ali_info" | awk '{print $NF}')
        echo "  AreTomo output:   ${bnx} × ${bny} × ${bnz} (binned at ${BINNING_FACTOR}x, from $(basename "$(dirname "$ali_sample")"))"
    fi
fi

# Show WARP settings TOMO_DIMS for comparison
if [ -f "$TILTSERIES_SETTINGS" ]; then
    wx=$(sed -n 's/.*Name="DimensionsX".*Value="\([0-9.]*\)".*/\1/p' "$TILTSERIES_SETTINGS" | head -1)
    wy=$(sed -n 's/.*Name="DimensionsY".*Value="\([0-9.]*\)".*/\1/p' "$TILTSERIES_SETTINGS" | head -1)
    wz=$(sed -n 's/.*Name="DimensionsZ".*Value="\([0-9.]*\)".*/\1/p' "$TILTSERIES_SETTINGS" | head -1)
    if [ -n "$wx" ] && [ -n "$wy" ] && [ -n "$wz" ]; then
        echo "  WARP settings:    ${wx} × ${wy} × ${wz} (unbinned, from warp_tiltseries.settings)"
        if [ -n "$bnx" ]; then
            expected_unbinned_x=$((bnx * BINNING_FACTOR))
            expected_unbinned_y=$((bny * BINNING_FACTOR))
            expected_unbinned_z=$((bnz * BINNING_FACTOR))
            if [ "$wx" = "$expected_unbinned_x" ] && [ "$wy" = "$expected_unbinned_y" ] && [ "$wz" = "$expected_unbinned_z" ]; then
                pass "WARP settings match AreTomo output (Phase 3.5 done ✓)"
            else
                warn "WARP settings ($wx × $wy × $wz) ≠ AreTomo × bin ($expected_unbinned_x × $expected_unbinned_y × $expected_unbinned_z) — run Phase 3.5!"
            fi
        fi
    fi
fi

if need_species; then
    echo "  DIAMETER:         $DIAMETER Å"
    echo "  TM_BOX_SIZE:      $TM_BOX_SIZE px (at ALIGN_ANGPIX)"
    echo "  SUBTOMO_BOX_SIZE: $SUBTOMO_BOX_SIZE px (at OUTPUT_ANGPIX)"
    echo "  TEMPLATERES:      $TEMPLATERES px"
fi

# CTF_RANGE_MAX ≥ 2×ANGPIX
if is_number "$ANGPIX" && is_number "$CTF_RANGE_MAX"; then
    min_ctf=$(awk "BEGIN {printf \"%.2f\", 2 * $ANGPIX}")
    if awk "BEGIN {exit !($CTF_RANGE_MAX >= $min_ctf)}"; then
        pass "CTF_RANGE_MAX=$CTF_RANGE_MAX ≥ 2×ANGPIX=$min_ctf"
    else
        fail "CTF_RANGE_MAX=$CTF_RANGE_MAX < 2×ANGPIX=$min_ctf"
    fi
fi

# Box sanity (species vars only)
if need_species; then
    if need_phase 6 && is_number "$TM_BOX_SIZE" && is_number "$ALIGN_ANGPIX" && is_number "$DIAMETER"; then
        tm_box_ang=$(awk "BEGIN {printf \"%.0f\", $TM_BOX_SIZE * $ALIGN_ANGPIX}")
        if awk "BEGIN {exit !($tm_box_ang >= $DIAMETER)}"; then
            pass "TM box covers particle: ${tm_box_ang}Å ≥ ${DIAMETER}Å"
        else
            fail "TM box (${tm_box_ang}Å) smaller than particle (${DIAMETER}Å)"
        fi
    fi
    if is_number "$SUBTOMO_BOX_SIZE" && is_number "$OUTPUT_ANGPIX" && is_number "$DIAMETER"; then
        sub_box_ang=$(awk "BEGIN {printf \"%.0f\", $SUBTOMO_BOX_SIZE * $OUTPUT_ANGPIX}")
        if awk "BEGIN {exit !($sub_box_ang >= $DIAMETER)}"; then
            pass "Subtomo box covers particle: ${sub_box_ang}Å ≥ ${DIAMETER}Å"
        else
            fail "Subtomo box (${sub_box_ang}Å) smaller than particle (${DIAMETER}Å)"
        fi
    fi
    if [ -n "$TEMPLATERES" ] && [ -n "$SUBTOMO_BOX_SIZE" ] && [ "$TEMPLATERES" != "$SUBTOMO_BOX_SIZE" ]; then
        warn "TEMPLATERES=$TEMPLATERES ≠ SUBTOMO_BOX_SIZE=$SUBTOMO_BOX_SIZE"
    fi
fi

# =============================================================================
# 7. Phase-completion checks — per-tomostar verification
# =============================================================================
banner "7. Phase completion status"

# Helper: for each tomostar, check that required outputs exist
check_per_ts() {
    local phase="$1" desc="$2"; shift 2
    local ts_ok=0 ts_total=0 ts_missing=0

    # Count tomostar files first for the progress bar
    local ts_count=0
    for ts_file in "$TOMOSTAR_DIR"/*.tomostar; do
        [ -f "$ts_file" ] && ((ts_count++))
    done

    if [ "$ts_count" -eq 0 ]; then
        info "Phase $phase: no tomostar files to check ($desc)"
        return
    fi

    local ts_idx=0
    for ts_file in "$TOMOSTAR_DIR"/*.tomostar; do
        [ ! -f "$ts_file" ] && continue
        ((ts_total+=1))
        ((ts_idx+=1))

        if [ "$PROGRESS_TTY" -eq 1 ]; then
            progress_bar "$ts_idx" "$ts_count" "$desc"
        fi

        local ts_name=$(basename "$ts_file" .tomostar)
        local ok=1
        for pattern in "$@"; do
            # pattern can contain __TS__ which gets replaced with the ts_name
            local expanded="${pattern//__TS__/$ts_name}"
            if ! compgen -G "$expanded" >/dev/null 2>&1; then
                ok=0
                break
            fi
        done
        if [ "$ok" -eq 1 ]; then
            ((ts_ok+=1))
        else
            ((ts_missing+=1))
        fi
    done

    # Clear progress bar before printing result
    if [ "$PROGRESS_TTY" -eq 1 ]; then
        progress_clear
    fi

    if [ "$ts_ok" -eq "$ts_total" ]; then
        pass "Phase $phase done: $ts_ok/$ts_total tomograms ($desc)"
    elif [ "$ts_ok" -gt 0 ]; then
        warn "Phase $phase partial: $ts_ok/$ts_total tomograms ($desc) — $ts_missing missing"
    else
        info "Phase $phase not yet run: 0/$ts_total tomograms ($desc)"
    fi
}

# Phase 1: frame series CTF — at least one .xml per frameseries
check_phase_output() {
    local phase="$1" desc="$2"; shift 2
    local found=0 total=0
    for pattern in "$@"; do
        ((total+=1))
        if compgen -G "$pattern" >/dev/null 2>&1; then
            ((found+=1))
        fi
    done
    if [ "$total" -gt 0 ]; then
        if [ "$found" -eq "$total" ]; then
            pass "Phase $phase output found ($desc)"
        elif [ "$found" -gt 0 ]; then
            warn "Phase $phase partially done: $found/$total outputs found ($desc)"
        else
            info "Phase $phase not yet run ($desc)"
        fi
    fi
}

if [ "$DRY_RUN" != "1" ]; then
    check_phase_output 1 "frame series CTF" \
        "$FRAMESERIES_DIR"/*.xml

    check_phase_output 2 "tomostar files" \
        "$TOMOSTAR_DIR"/*.tomostar

    # Phase 3: per-tomostar — _ali.mrc, .tlt, .xf for each tomostar
    check_per_ts 3 "AreTomo output" \
        "$TILTSTACK_DIR/__TS__/__TS___ali.mrc" \
        "$TILTSTACK_DIR/__TS__/__TS__.tlt" \
        "$TILTSTACK_DIR/__TS__/__TS__.xf"

    # Phase 5: per-tomostar — WARP reconstruction for each tomostar
    check_per_ts 5 "WARP reconstruction" \
        "$WORK_DIR/warp_tiltseries/reconstruction/__TS__"_*Apx.mrc

    # After Phase 5, remind user to compare AreTomo vs WARP dims
    if need_phase 5 || need_phase 7 || [ -z "$PHASE" ]; then
        phase3_ok=0; phase5_ok=0; ts_total=0
        # Pre-count for progress bar
        ts_cmp_count=0
        for ts_file in "$TOMOSTAR_DIR"/*.tomostar; do
            [ -f "$ts_file" ] && ((ts_cmp_count++))
        done
        ts_cmp_idx=0
        for ts_file in "$TOMOSTAR_DIR"/*.tomostar; do
            [ ! -f "$ts_file" ] && continue
            ((ts_total++))
            ((ts_cmp_idx++))
            if [ "$PROGRESS_TTY" -eq 1 ]; then
                progress_bar "$ts_cmp_idx" "$ts_cmp_count" "Comparing Phase 3/5 dims"
            fi
            ts_name=$(basename "$ts_file" .tomostar)
            if compgen -G "$TILTSTACK_DIR/$ts_name/${ts_name}_ali.mrc" >/dev/null 2>&1 && \
               compgen -G "$TILTSTACK_DIR/$ts_name/${ts_name}.tlt" >/dev/null 2>&1 && \
               compgen -G "$TILTSTACK_DIR/$ts_name/${ts_name}.xf" >/dev/null 2>&1; then
                ((phase3_ok++))
            fi
            if compgen -G "$WORK_DIR/warp_tiltseries/reconstruction/${ts_name}_"*Apx.mrc >/dev/null 2>&1; then
                ((phase5_ok++))
            fi
        done
        if [ "$PROGRESS_TTY" -eq 1 ] && [ "$ts_cmp_count" -gt 0 ]; then
            progress_clear
        fi
        if [ "$ts_total" -gt 0 ] && [ "$phase3_ok" -eq "$ts_total" ] && [ "$phase5_ok" -eq "$ts_total" ]; then
            echo ""
            yellow "  ⚠ Both Phase 3 and Phase 5 complete — visually verify they match:"
            echo "     headerPyTom warp_tiltseries/tiltstack/TS_XXX/TS_XXX_ali.mrc"
            echo "     headerPyTom warp_tiltseries/reconstruction/TS_XXX_*Apx.mrc"
            echo "     Both must have identical nx × ny × nz (both at ALIGN_ANGPIX),"
            echo "     and similar reconstruction at every slice."
            echo "     Mismatch → Phase 3.5 skipped or --alignment_angpix wrong in Phase 4."
        fi
    fi

    if need_species; then
        check_phase_output "6a" "template" \
            "$TEMPLATE_MRC"
        check_phase_output "6b" "sphere mask" \
            "$TM_MASK_MRC"

        # Phase 6c: per-tomostar — job.xml for each tomostar
        check_per_ts "6c" "TM job XMLs" \
            "$TM_JOBS_DIR/__TS__/job.xml"

        # Phase 6d: per-tomostar — scores and angles for each tomostar
        check_per_ts "6d" "TM scores+angles" \
            "$TM_JOBS_DIR/__TS__/scores_*.em" \
            "$TM_JOBS_DIR/__TS__/angles_*.em"

        # Phase 6e: per-tomostar — extracted particles
        check_per_ts "6e" "extracted particles" \
            "$TM_PARTICLES_DIR/__TS___particles.xml"

        # Phase 6f: per-tomostar — STAR conversion
        check_per_ts "6f" "STAR files" \
            "$TM_STAR_DIR/__TS__.star"

        # Phase 6g: per-tomostar — WARP-compatible STAR
        check_per_ts "6g" "WARP STAR files" \
            "$TM_WARP_DIR/__TS___warp.star"

        # Phase 7: per-tomostar — exported subtomograms + combined STAR
        check_per_ts 7 "exported subtomograms" \
            "$DATADIR/__TS__/"*.mrc
        check_phase_output 7 "combined export STAR" \
            "$EXPORT_STAR"

        check_phase_output "8a" "training mask" \
            "$TRAINING_MASK_MRC"

        check_phase_output "8b" "training output" \
            "$WORK_DIR/$OUTPUT_DIR"/weights.*.pkl
    fi
else
    info "Skipped in --dry-run mode"
fi

# =============================================================================
# 8. Dry-run command summary
# =============================================================================
if [ "$DRY_RUN" = "1" ]; then
    banner "8. Dry-run — Phase ${PHASE:-all}"
    echo ""
    echo "  ── Configuration ──"
    echo "  pipeline.conf:    $PIPELINE_CONF"
    need_species && echo "  species conf:     $SPECIES_CONF"
    echo ""
    echo "  ── Resolved paths ──"
    echo "  WORK_DIR:           $WORK_DIR"
    echo "  WARP_DIR:           $WARP_DIR"
    echo "  ANGPIX:             $ANGPIX Å | BINNING: ${BINNING_FACTOR}x | ALIGN_ANGPIX: $ALIGN_ANGPIX Å"
    need_species && echo "  TM_LABEL:           $TM_LABEL"
    need_species && echo "  DIAMETER:           $DIAMETER Å"
    need_species && echo "  TM_BOX_SIZE:        $TM_BOX_SIZE px (at ALIGN_ANGPIX)"
    need_species && echo "  SUBTOMO_BOX_SIZE:   $SUBTOMO_BOX_SIZE px (at OUTPUT_ANGPIX=$OUTPUT_ANGPIX)"
    need_species && echo "  TEMPLATERES:        $TEMPLATERES px"
    need_species && echo "  TEMPLATE_MRC:       $TEMPLATE_MRC"
    need_species && echo "  TM_MASK_MRC:        $TM_MASK_MRC"
    need_species && echo "  TRAINING_MASK_MRC:  $TRAINING_MASK_MRC"
    need_species && echo "  EXPORT_STAR:        $EXPORT_STAR"
    need_species && echo "  DATADIR:            $DATADIR"
    need_species && echo "  OPUSET_OUT_DIR:     $OPUSET_OUT_DIR"
    echo ""
    echo "  ── Tilt series (from tomostar) ──"
    ts_count=0
    for ts_file in "$TOMOSTAR_DIR"/*.tomostar; do
        [ ! -f "$ts_file" ] && continue
        ((ts_count+=1))
    done
    echo "  Count: $ts_count"
    if [ "$ts_count" -gt 0 ] && [ "$ts_count" -le 5 ]; then
        for ts_file in "$TOMOSTAR_DIR"/*.tomostar; do
            [ ! -f "$ts_file" ] && continue
            echo "    - $(basename "$ts_file" .tomostar)"
        done
    fi
    echo ""
    echo "  ── Slurm submit commands ──"
    for s in $(phase_scripts "$PHASE"); do
        echo "  sbatch --export=ALL,SKILL_DIR=\"\$(pwd)\" scripts/$s"
    done
    echo ""
    echo "  ── Key command preview ──"
    case "${PHASE:-all}" in
        3a) echo "  WarpTools ts_stack --angpix $ALIGN_ANGPIX" ;;
        3b) echo "  AreTomo2 -InMrc TS_XXX.st -OutMrc TS_XXX_ali.mrc -VolZ $((TOMO_DIM_Z / BINNING_FACTOR)) -AlignZ $((TOMO_DIM_Z / BINNING_FACTOR * 40 / 100)) -AngFile TS_XXX_neg.rawtlt" ;;
        3.5)
            ali_sample=$(find "$TILTSTACK_DIR" -name "*_ali.mrc" -type f -print -quit 2>/dev/null)
            if [ -n "$ali_sample" ]; then
                ali_info=$(conda run -n "$PYTOM_ENV" headerPyTom "$ali_sample" 2>/dev/null | grep 'Number of columns' || true)
                if [ -n "$ali_info" ]; then
                    bnx=$(echo "$ali_info" | awk '{print $(NF-2)}')
                    bny=$(echo "$ali_info" | awk '{print $(NF-1)}')
                    bnz=$(echo "$ali_info" | awk '{print $NF}')
                    ux=$((bnx * BINNING_FACTOR)); uy=$((bny * BINNING_FACTOR)); uz=$((bnz * BINNING_FACTOR))
                    echo "  WarpTools create_settings --tomo_dimensions ${ux}x${uy}x${uz} --output warp_tiltseries.settings"
                    echo "    (derived from AreTomo ${bnx}x${bny}x${bnz} × BINNING=$BINNING_FACTOR)"
                else
                    echo "  WarpTools create_settings --tomo_dimensions <from _ali.mrc × $BINNING_FACTOR> --output warp_tiltseries.settings"
                fi
            else
                echo "  WarpTools create_settings --tomo_dimensions <from _ali.mrc × $BINNING_FACTOR> --output warp_tiltseries.settings"
            fi
            ;;
        3)   echo "  WarpTools ts_stack --angpix $ALIGN_ANGPIX"; echo "  AreTomo2 ... -AngFile TS_XXX_neg.rawtlt" ;;
        4)   echo "  WarpTools ts_import_alignments --alignment_angpix $ALIGN_ANGPIX" ;;
        5a)  echo "  WarpTools ts_defocus_hand --set_auto && WarpTools ts_ctf --range_high $CTF_RANGE_MAX" ;;
        5b)  echo "  WarpTools ts_reconstruct --angpix $ALIGN_ANGPIX" ;;
        5)   echo "  WarpTools ts_defocus_hand --set_auto && WarpTools ts_ctf --range_high $CTF_RANGE_MAX"; echo "  WarpTools ts_reconstruct --angpix $ALIGN_ANGPIX" ;;
        6a)  echo "  create_template.py -f '$INPUT_MRC' -d '$TEMPLATES_DIR' -o '${TM_LABEL}_tm.mrc' -s $ANGPIX --map-spacing $MAP_ANGPIX -b $BINNING_FACTOR -x $TM_BOX_SIZE" ;;
        6b)
            if [ -f "$TEMPLATE_MRC" ]; then
                tpl_info=$(conda run -n "$PYTOM_ENV" headerPyTom "$TEMPLATE_MRC" 2>/dev/null | grep 'Number of columns' || true)
                if [ -n "$tpl_info" ]; then
                    tdim=$(echo "$tpl_info" | awk '{print $(NF-2)}')
                    echo "  create_mask.py -o '$TM_MASK_MRC' -b $tdim -r $MASK_RADIUS -s ${MASK_SIGMA:-1}"
                else
                    echo "  create_mask.py -o '$TM_MASK_MRC' -b <template_dim> -r $MASK_RADIUS -s ${MASK_SIGMA:-1}"
                fi
            else
                echo "  create_mask.py -o '$TM_MASK_MRC' -b <from template> -r $MASK_RADIUS -s ${MASK_SIGMA:-1}"
            fi
            ;;
        6c)  echo "  Loop tomostar/*.tomostar → job.xml per TS_XXX (gen_tm_jobs_aretomo.slurm)" ;;
        6d)  echo "  mpiexec -n 4 localization.py -j job.xml -x ${TM_SPLIT_X:-2} -y ${TM_SPLIT_Y:-2} -z ${TM_SPLIT_Z:-1} --gpuID 0,1,2,3   (per TS_XXX)" ;;
        6e)  echo "  extractCandidates.py --tm-dir ... --num-candidates $NUM_CANDIDATES --mask-radius ${EXTRACT_MASK_RADIUS:-14} --template $TEMPLATE"; echo "    → reads scores_${TEMPLATE}.em / angles_${TEMPLATE}.em per TS" ;;
        6f)  echo "  convert.py -f TS_XXX_particles.xml --outname TS_XXX.star --pixelSize $ANGPIX --binPyTom $BINNING_FACTOR -o star" ;;
        6g)  echo "  dsdsh convert_pytom TS_XXX.star TS_XXX   (→ TS_XXX_warp.star)" ;;
        6)
            echo "  create_template.py -f '$INPUT_MRC' -d '$TEMPLATES_DIR' -o '${TM_LABEL}_tm.mrc' -s $ANGPIX --map-spacing $MAP_ANGPIX -b $BINNING_FACTOR -x $TM_BOX_SIZE"
            echo "  create_mask.py -o '$TM_MASK_MRC' -b <template_dim> -r $MASK_RADIUS -s ${MASK_SIGMA:-1}"
            echo "  gen_tm_jobs_aretomo.slurm → job.xml per TS_XXX"
            echo "  mpiexec -n 4 localization.py -j job.xml -x ${TM_SPLIT_X:-2} -y ${TM_SPLIT_Y:-2} -z ${TM_SPLIT_Z:-1} --gpuID 0,1,2,3   (per TS_XXX)"
            echo "  extractCandidates.py --tm-dir ... --num-candidates $NUM_CANDIDATES --mask-radius ${EXTRACT_MASK_RADIUS:-14} --template $TEMPLATE"
            echo "  convert.py -f TS_XXX_particles.xml --outname TS_XXX.star --pixelSize $ANGPIX --binPyTom $BINNING_FACTOR -o star"
            echo "  dsdsh convert_pytom TS_XXX.star TS_XXX   (→ TS_XXX_warp.star)"
            ;;
        7)
            echo "  WarpTools ts_export_particles --output_angpix $OUTPUT_ANGPIX --box $SUBTOMO_BOX_SIZE --diameter $DIAMETER --coords_angpix $COORDS_ANGPIX"
            echo "    → $EXPORT_STAR"
            echo "    → $DATADIR/TS_XXX/*.mrc"
            ;;
        8a)
            echo "  dsdsh create_mask (sample subtomo from DATADIR) -o '$TRAINING_MASK_MRC'"
            echo "    --sphere-radius ${SPHERE_RADIUS:-<auto>} --soft-edge $MASK_SOFT_EDGE"
            ;;
        8b)
            echo "  torchrun --nproc_per_node=$NUM_GPUS -m cryodrgn.commands.train_tomo_dist \\"
            echo "    '$EXPORT_STAR' --poses ${POSE_PKL:-<auto>} -n $NUM_EPOCHS -b $BATCH_SIZE \\"
            echo "    --zdim $ZDIM --zaffinedim $ZAFFINEDIM --lr $LEARNING_RATE \\"
            echo "    --num-gpus $NUM_GPUS --multigpu --beta-control $BETA_CONTROL \\"
            echo "    -o '$OPUSET_OUT_DIR' -r '$TRAINING_MASK_MRC' --templateres $TEMPLATERES \\"
            echo "    --datadir '$(dirname $DATADIR)' --angpix ${ANGPIX:-(auto)} --downfrac ${DOWNSAMPLE_FRAC:-1.} --warp --tilt-range ${TILT_RANGE:-(auto)} --tilt-step ${TILT_STEP:-(auto)}"
            ;;
        8c)
            echo "  torchrun --nproc_per_node=$NUM_GPUS -m cryodrgn.commands.train_tomo_dist \\"
            echo "    ... --encode-mode fixed --valfrac 0.0 → '$WORK_DIR/opuset/$TM_LABEL/fixed_subset1/'"
            ;;
        8)
            echo "  dsdsh create_mask (sample subtomo from DATADIR) -o '$TRAINING_MASK_MRC' --sphere-radius ${SPHERE_RADIUS:-<auto>} --soft-edge $MASK_SOFT_EDGE"
            echo "  torchrun --nproc_per_node=$NUM_GPUS -m cryodrgn.commands.train_tomo_dist --encode-mode grad ..."
            echo "  torchrun --nproc_per_node=$NUM_GPUS -m cryodrgn.commands.train_tomo_dist --encode-mode fixed ..."
            ;;
    esac
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
banner "Validation Summary"
echo ""
[ -n "$PHASE" ] && echo "  Phase:    $PHASE" || echo "  Phase:    all"
if need_species; then
    echo "  Species:  $(basename "$SPECIES_CONF")"
fi
echo "  Errors:   $ERRORS"
echo "  Warnings: $WARNINGS"
echo ""

if [ "$ERRORS" -eq 0 ]; then
    green "All checks passed."
    if [ -n "$PHASE" ]; then
        echo ""
        echo "Next: submit Phase $PHASE:"
        for s in $(phase_scripts "$PHASE"); do
            echo "  sbatch --export=ALL,SKILL_DIR=\"\$(pwd)\" scripts/$s"
        done
    else
        echo ""
        echo "Next: submit Phase 1:"
        echo "  sbatch --export=ALL,SKILL_DIR=\"\$(pwd)\" scripts/warp_frameseries_import.slurm"
    fi
else
    red "Fix $ERRORS error(s) before submitting."
fi

exit $ERRORS
