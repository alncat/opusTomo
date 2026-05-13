# WARP/OPUS-ET Pipeline — Phase-by-Phase Commands

Detailed commands for each phase. Use variables defined in SKILL.md Quick Start.

---

## Phase 1: Frame Series Processing

### Determine frame format + type

```bash
ls "$FRAME_DIR" | head -5           # check extension first
# *.eer  → Falcon 4 EER   → Option C
# *.tif  → TIFF movie     → Option B (tiff variant)
# *.mrc  → inspect Z dimension via one of:

# Preferred — PyTOM env is already required:
conda activate "$PYTOM_ENV"
headerPyTom <any_input_file.mrc>    # look at "Number of columns, rows, sections"

# Alternative — only if IMOD is on PATH:
header <any_input_file.mrc>

# Fallback — works in any env with mrcfile (e.g. WARP_ENV):
python -c "import mrcfile,sys; print(mrcfile.open(sys.argv[1],permissive=True).data.shape)" <file.mrc>

# Z > 1 → fractionated (Option B/C);  Z == 1 → single frame (Option A)
```
- Don't assume `header` (IMOD) is installed — the skill does **not** declare IMOD as a dependency. Default to `headerPyTom`.
- Filename patterns and MDOC `NumberOfFrames` are not reliable.

### Option A: Single-frame MRC (no motion correction)

```bash
WarpTools create_settings \
    --folder_data "$FRAME_DIR" \
    --folder_processing warp_frameseries \
    --extension "*.mrc" \
    --angpix $ANGPIX \
    --exposure $EXPOSURE \
    --output warp_frameseries.settings

mkdir -p warp_frameseries/average
for f in "$FRAME_DIR"/*.mrc; do
    ln -sf "$(realpath "$f")" "warp_frameseries/average/$(basename "$f")"
done

WarpTools fs_ctf \
    --settings warp_frameseries.settings \
    --grid 2x2x1 \
    --range_max $CTF_RANGE_MAX \
    --defocus_max 8 \
    --perdevice 1
```
No gain reference — assumed already applied to the averaged image.

### Option B: Fractionated MRC / TIFF (motion correction + CTF)

```bash
# Set extension per format: "*fractions.mrc" | "*.mrc" | "*.tif" | "*.tiff"
WarpTools create_settings \
    --folder_data "$FRAME_DIR" \
    --folder_processing warp_frameseries \
    --extension "$FRAME_EXT" \
    --angpix $ANGPIX \
    --exposure $EXPOSURE \
    --bin 1 \
    --gain_path "$GAIN_FILE" \
    $( [ -n "$DEFECTS_FILE" ] && echo "--defects_path $DEFECTS_FILE" ) \
    --output warp_frameseries.settings

export WARP_FORCE_MRC_FLOAT32=1
WarpTools fs_motion_and_ctf \
    --settings warp_frameseries.settings \
    --m_grid 1x1x3 \
    --c_grid 2x2x1 \
    --c_range_max $CTF_RANGE_MAX \
    --c_defocus_max 8 \
    --c_use_sum \
    --out_averages \
    --out_average_halves \
    --perdevice 1
```
- Omit `--gain_path` if `GAIN_FILE=""`.
- Gain orientation: add `--gain_flip_x`, `--gain_flip_y`, or `--gain_transpose` only if the user explicitly provides them; otherwise let WARP's defaults apply.

### Option C: EER (Falcon 4) — fractionated with frame grouping

```bash
WarpTools create_settings \
    --folder_data "$FRAME_DIR" \
    --folder_processing warp_frameseries \
    --extension "*.eer" \
    --angpix $ANGPIX \
    --exposure $EXPOSURE \
    --bin 1 \
    --gain_path "$GAIN_FILE" \
    --eer_ngroups $EER_NGROUPS \
    $( [ -n "$DEFECTS_FILE" ] && echo "--defects_path $DEFECTS_FILE" ) \
    --output warp_frameseries.settings

export WARP_FORCE_MRC_FLOAT32=1
WarpTools fs_motion_and_ctf \
    --settings warp_frameseries.settings \
    --m_grid 1x1x3 \
    --c_grid 2x2x1 \
    --c_range_max $CTF_RANGE_MAX \
    --c_defocus_max 8 \
    --c_use_sum \
    --out_averages \
    --out_average_halves \
    --perdevice 1
```
- `--eer_ngroups` merges raw EER frames into `N` dose fractions. Typical: 10–20.
- Gain reference is almost always required for EER; ask the user.

### Mixed formats in one session
`create_settings` accepts a single `--extension` pattern. If a session mixes EER and MRC (or different gain references), split into separate frame-series subdirectories and run Phase 1 once per subset, then merge tomostar listings in Phase 2.

**SLURM:** `sbatch scripts/warp_frameseries_import.slurm` — the template covers Option A/B; for EER (Option C) or custom gain flags, add the extra flags in its CONFIGURATION section.

---

## Phase 2: Tilt Series Import

```bash
# 1. Import tilt series from MDOC (creates tomostar files)
WarpTools ts_import \
    --mdocs mdoc/ \
    --frameseries warp_frameseries \
    --tilt_exposure $EXPOSURE \
    --min_intensity 0.3 \
    --output tomostar

# 2. Create tilt series settings (can run before or after ts_import)
WarpTools create_settings \
    --folder_data tomostar \
    --folder_processing warp_tiltseries \
    --extension "*.tomostar" \
    --angpix $ANGPIX \
    --exposure $EXPOSURE \
    --tomo_dimensions $TOMO_DIMS \
    --output warp_tiltseries.settings
```

**SLURM:** `sbatch scripts/warp_tiltseries_setup.slurm`

**Note:** `create_settings` only writes a config file pointing to the tomostar directory — it does not require the tomostar files to exist yet. The SLURM script runs `create_settings` first, then `ts_import`.

**Initial `TOMO_DIMS` is provisional.** The Z dimension in particular is often a guess (e.g. `2000`). Don't worry about getting it exactly right here — Phase 3.5 rewrites it from the actual AreTomo output before downstream phases read it.

---

## Phase 3: Tilt Stack Export + AreTomo2 Alignment

```bash
# Export tilt stacks (4x binned = angpix × 4)
WarpTools ts_stack \
    --settings warp_tiltseries.settings \
    --angpix $ALIGN_ANGPIX \
    --perdevice 1
```
**SLURM:** `sbatch scripts/warp_export_stacks.slurm`

### AreTomo2 — Angle Negation Method (recommended)

WARP writes angles +max → -min (e.g. +54…-54); AreTomo expects -max → +min (e.g. -54…+54). The negation script handles this automatically:

```bash
sbatch scripts/warp_aretomo_align_negate.slurm
```

What it does:
1. Negates `.rawtlt` → `_neg.rawtlt` (WARP → AreTomo convention)
2. Runs AreTomo2 with negated angles
3. Negates output `.tlt` back (AreTomo → WARP convention)
4. Copies `.xf` transform file unchanged

### Manual AreTomo2 (single tilt series)
```bash
AreTomo2 \
    -InMrc TS_026.st \
    -OutMrc TS_026_ali.mrc \
    -OutBin 1 \
    -AngFile TS_026_neg.rawtlt \
    -OutImod 2 \
    -VolZ $BINNED_Z \
    -Gpu 0 \
    -AlignZ $ALIGN_Z \
    -FlipVol 1 \
    -FlipInt 1 \
    -DarkTol 0.1 \
    -Wbp 1
```

---

## Phase 3.5: Update TOMO_DIMS from AreTomo Output (REQUIRED)

Phase 3 does not depend on Phase 2's `TOMO_DIMS` — AreTomo reads the tilt stack directly. But Phase 4 and everything downstream read `warp_tiltseries.settings`, so the settings must reflect the true **unbinned** tomogram size before proceeding.

```bash
# Read binned dims from any AreTomo output
headerPyTom warp_tiltseries/tiltstack/TS_XXX/TS_XXX_ali.mrc
# size: BX BY BZ   ← take these three values

# Compute unbinned dims
unbinned_x=$((BX * BINNING_FACTOR))
unbinned_y=$((BY * BINNING_FACTOR))
unbinned_z=$((BZ * BINNING_FACTOR))
NEW_TOMO_DIMS="${unbinned_x}x${unbinned_y}x${unbinned_z}"

# Re-run create_settings with corrected dims
# Preserve ANGPIX from the existing warp_tiltseries.settings PixelSize.
ANGPIX=$(sed -n 's/.*Name="PixelSize".*Value="\([0-9.][0-9.]*\)".*/\1/p' warp_tiltseries.settings | head -1)
WarpTools create_settings \
    --folder_data tomostar \
    --folder_processing warp_tiltseries \
    --extension "*.tomostar" \
    --angpix $ANGPIX \
    --exposure $EXPOSURE \
    --tomo_dimensions "$NEW_TOMO_DIMS" \
    --output warp_tiltseries.settings
```

**SLURM:** `sbatch scripts/warp_update_tomo_dims.slurm` — handles the header parse, binning multiply, and settings rewrite in one job. Backs up the previous settings as `warp_tiltseries.settings.before_update`.

**Why this matters:** the unbinned `TOMO_DIMS` in `warp_tiltseries.settings`
defines the coordinate frame used by all later WARP commands; if it doesn't
match the actual reconstruction, downstream coordinates will be wrong.

---

## Phase 4: Import Alignment Parameters

```bash
WarpTools ts_import_alignments \
    --settings warp_tiltseries.settings \
    --alignments warp_tiltseries/tiltstack/ \
    --alignment_angpix $ALIGN_ANGPIX
```
**SLURM:** `sbatch scripts/warp_import_alignments.slurm`

**Critical:** Use the exact same pixel size as the AreTomo export. The `.xf` and `.tlt` files are in binned pixel units. The SLURM script derives `ALIGN_ANGPIX = ANGPIX × BINNING_FACTOR`; keep `BINNING_FACTOR` identical to `warp_export_stacks.slurm` / AreTomo, and leave `ANGPIX` blank to read `PixelSize` from `warp_tiltseries.settings`.

---

## Phase 5: CTF Estimation + Reconstruction

```bash
# Step 1: Check defocus handedness
WarpTools ts_defocus_hand \
    --settings warp_tiltseries.settings \
    --check
# If output says "average correlation is negative":
WarpTools ts_defocus_hand \
    --settings warp_tiltseries.settings \
    --set_flip

# Step 2: CTF estimation
WarpTools ts_ctf \
    --settings warp_tiltseries.settings \
    --window 512 \
    --range_high 7 \
    --defocus_max 8 \
    --perdevice 1

# Step 3: Reconstruct tomograms (32-bit float)
export WARP_FORCE_MRC_FLOAT32=1
WarpTools ts_reconstruct \
    --settings warp_tiltseries.settings \
    --angpix $ALIGN_ANGPIX \
    --perdevice 1
```

The SLURM script derives `ALIGN_ANGPIX = ANGPIX × BINNING_FACTOR` the same way as Phase 4. Do not hardcode a dataset-specific binned pixel size here.

### Sanity check: AreTomo vs. WARP reconstruction dims must match

After `ts_reconstruct`, **always verify** that the WARP tomogram has the same binned size as the AreTomo `_ali.mrc`. Both are reconstructed at `ALIGN_ANGPIX`, so their `nx × ny × nz` should be identical. A mismatch means **either** the alignment import failed (Phase 4) **or** `TOMO_DIMS` in the settings is wrong (Phase 3.5 was skipped or used the wrong binning).

```bash
# Pick any tilt series
TS=TS_026

# AreTomo reconstruction (binned, from Phase 3)
headerPyTom warp_tiltseries/tiltstack/$TS/${TS}_ali.mrc

# WARP reconstruction (binned at ALIGN_ANGPIX, from Phase 5; named with pixel-size suffix)
headerPyTom warp_tiltseries/reconstruction/${TS}_*Apx.mrc

# Both `size:` lines should report the same NX × NY × NZ.
```

If they differ:
- **Z mismatch** → Phase 3.5 wasn't run, or `BINNING_FACTOR` in `warp_update_tomo_dims.slurm` doesn't match the AreTomo binning. Re-run Phase 3.5 and Phase 4.
- **X/Y mismatch** → `TOMO_DIMS` X or Y in settings was wrong. Re-run `create_settings` with the correct unbinned detector size, then Phase 4.
- **Same size but content looks shifted/cropped** → `--alignment_angpix` in Phase 4 didn't match `ALIGN_ANGPIX` from `ts_stack`. Re-run Phase 4 with the correct value.

**SLURM (step-by-step):**
```bash
sbatch scripts/warp_ts_ctf.slurm         # Steps 1+2: handedness + CTF
sbatch scripts/warp_ts_reconstruct.slurm # Step 3: reconstruction
```

### End-to-end pixel-size audit

Most "particles look wrong" failures trace back to one pixel-size or binning
mismatch carried silently across phases. Before running Phase 6 (TM), verify
that every stage's pixel size matches the value you expect:

```bash
TS=TS_026   # any tilt series

echo "ANGPIX (gather, unbinned):       $ANGPIX"
echo "BINNING_FACTOR:                  $BINNING_FACTOR"
echo "ALIGN_ANGPIX (= ANGPIX × bin):   $ALIGN_ANGPIX"
echo ""
echo "AreTomo _ali.mrc:"
headerPyTom warp_tiltseries/tiltstack/$TS/${TS}_ali.mrc | grep -E 'columns|spacing'
echo ""
echo "WARP recon:"
headerPyTom warp_tiltseries/reconstruction/${TS}_*Apx.mrc | grep -E 'columns|spacing'
echo ""
echo "Template:"
headerPyTom $TEMPLATE_MRC | grep -E 'columns|spacing'
```

**What to expect:**
- AreTomo `_ali.mrc` pixel spacing = `ALIGN_ANGPIX` (binned).
- WARP recon pixel spacing = `ALIGN_ANGPIX` (binned, set by `ts_reconstruct --angpix $ALIGN_ANGPIX`).
- Template pixel spacing = `ALIGN_ANGPIX` (TM runs on the binned recons; template *must* match).
- After Phase 7 export, exported subtomos' pixel spacing = `OUTPUT_ANGPIX` (user-chosen; this is the only one that legitimately differs).

If the template's spacing ≠ `ALIGN_ANGPIX`, **stop and resample the template** before Phase 6 — PyTom expects template and tomogram at the same pixel size.

---

## Phase 6: Template Matching

```bash
# 1. Generate sphere mask from template
sbatch scripts/gen_sphere_mask.slurm
# → mask_ribo.mrc

# 2. Generate PyTom job XMLs (uses tomostar as canonical tilt series list)
sbatch scripts/gen_tm_jobs_aretomo.slurm
# → template_matching/jobs/TS_XXX_*/job.xml

# 3. Run template matching (sequential to avoid GPU OOM)
sbatch scripts/run_tm_sequential.slurm
# → scores_*.em, angles_*.em per job directory

# 4. Extract particle candidates (parallel)
sbatch scripts/extract_tm_candidates_parallel.slurm
# → particles/TS_XXX_particles.xml

# 5. Convert PyTom XML → per-tilt-series STAR (canonical SLURM path)
sbatch scripts/convert_to_star.slurm
# Uses PyTom convert.py per file. Fill ANGPIX and BINNING_FACTOR in the script:
# --pixelSize / ANGPIX: UNBINNED pixel size (coords are written unbinned).
# --binPyTom / BINNING_FACTOR: binning at which PyTom ran TM, same as AreTomo.
# Do not use a combined particles.star here.

# 6. Convert STAR → WARP format
sbatch scripts/convert_pytom_to_warp.slurm
# → warp_star/*_warp.star

# Optional interactive shortcut:
# dsdsh convert_pytom_to_star can batch XML files with --pixel-size and
# --bin-size, but keep its output per tilt series if you use it in this
# workflow; the canonical scripts above are the documented route.
```

---

## Phase 7: Export Subtomograms for OPUS-ET

```bash
# Export subtomograms (3D, with CTF metadata)
WarpTools ts_export_particles \
    --settings warp_tiltseries.settings \
    --input_directory warp_star/ \
    --input_pattern "*_warp.star" \
    --output_star warp_tiltseries/matchingribo.star \
    --output_angpix $OUTPUT_ANGPIX \
    --box $SUBTOMO_BOX_SIZE \
    --diameter $DIAMETER \
    --relative_output_paths \
    --3d \
    --coords_angpix $COORDS_ANGPIX \
    --output_ctf_csv \
    --dont_correct_ctf_3d

# OPUS-ET training scripts generate the pose pickle from this STAR automatically
# with dsd parse_pose_star. Do not run dsdsh prepare unless doing a custom,
# manual training setup.
```
**SLURM:** `sbatch scripts/warp_export_particles.slurm`

`OUTPUT_ANGPIX` is the chosen exported subtomogram pixel size and is often binned for training. `COORDS_ANGPIX` is the coordinate pixel size in the input STAR, usually unbinned `ANGPIX` after the PyTom→WARP conversion.

**Required flags for OPUS-ET (alncat WARP fork only):**
- `--output_ctf_csv` — OPUS-ET reads CTF from CSV, not corrected volumes
- `--dont_correct_ctf_3d` — OPUS-ET handles CTF correction internally

These two flags exist only in the alncat fork
(<https://github.com/alncat/warp>) on the `alncat` branch; they are not in
upstream WARP. Verify with
`WarpTools ts_export_particles --help | grep dont_correct_ctf_3d`. If empty,
build the fork before this phase:

```bash
git clone https://github.com/alncat/warp.git
cd warp
git checkout alncat
# Then follow upstream WARP build instructions; point WARP_DIR at the
# resulting Release/linux-x64/publish/.
```

---

## Phase 8: OPUS-ET Training

```bash
# Optional: generate training mask
sbatch scripts/gen_training_mask.slurm
# → mask_ribo_training.mrc

# Check tilt angles before training
head -1 warp_tiltseries/tiltstack/TS_XXX/TS_XXX.tlt  # max angle
head -3 warp_tiltseries/tiltstack/TS_XXX/TS_XXX.tlt  # step size
```

### Option A: Heterogeneity analysis (train_tomo_dist)
```bash
sbatch scripts/train_opuset.slurm
# Key params: ZDIM, ZAFFINEDIM, NUM_EPOCHS, OUTPUT_DIR, TILT_RANGE, TILT_STEP
```

After training:
```bash
dsdsh analyze $OUTPUT_DIR $((NUM_EPOCHS-1)) 10 20
```

### Option B: Subtomogram averaging (encode-mode fixed)
For homogeneous averaging — uses `--encode-mode fixed` to set z=0 for all particles.
**SLURM:** `sbatch scripts/train_opuset_fixed.slurm`. See `references/advanced.md`
for full command and parameters.

**Generating half maps for M refinement.** Split the Phase 7 STAR by
`rlnRandomSubset` and run fixed-mode once per subset:

```bash
dsdsh convert_star <Phase 7 STAR> $ANGPIX --subset-label 1
dsdsh convert_star <Phase 7 STAR> $ANGPIX --subset-label 2
```

The two resulting volumes are the `half1.mrc` / `half2.mrc` inputs to
`warp_m_create_species.slurm`. See `references/advanced.md` →
"Prerequisites: half maps from OPUS-ET fixed mode".
