---
name: warp-opus-et
description: "Cryo-ET data processing pipeline using WARP, AreTomo2, PyTOM, and OPUS-ET. Use this skill whenever the user is processing cryo-electron tomography (cryo-ET) data, including: importing frame series or tilt series from MDOC files, CTF estimation, AreTomo2 alignment, template matching with PyTOM, exporting subtomograms, or training OPUS-ET heterogeneity models. Also use when the user mentions tomostar files, WarpTools commands, tilt stacks (.st, .rawtlt), defocus handedness, ts_export_particles, or SLURM job submission for GPU tomography processing."
---

# WARP/OPUS-ET Cryo-ET Processing Workflow

## Agent Rules — read before acting

- **Do only what the user asks.** Don't anticipate, extend, or fix unreported issues — even obvious ones. One change at a time.
- **Read before editing.** Always read the current file/script before describing or modifying it. Don't rely on remembered content — the codebase changes.
- **Verify before asserting.** If uncertain how a tool, flag, or script behaves, check (`--help`, API docs, actual output) rather than inferring from naming. Wrong documentation is worse than no documentation.
- **Enumerate scope before acting.** For multi-file changes, use `grep` to find all affected files and confirm scope with the user before making any edits.
- **Show before and after for every edit.** Before modifying a script or config, quote the relevant current lines. After editing, summarize exactly what changed — not just "done."
- **Don't chain changes.** Renaming a variable in one place does NOT mean you should rename it everywhere — confirm scope first.
- **Don't redesign.** Apparent inconsistencies may be intentional. Respect the existing pattern unless the user asks to change it.
- **When in doubt, show the current state and ask.** "Here's what I see. Do you want me to change X, Y, or both?"

## IMPORTANT: Gather Experimental Settings First

**Before generating any commands**, collect the parameters below. Do NOT use placeholder values — wait for actual answers. Only ask about phases the user needs.

Try to determine frame type automatically before asking (see Frame Type below).

### Microscope & Detector
1. **Pixel size** (`ANGPIX`) — physical pixel size in Å (e.g., `1.085`, `2.17`, `3.37`)
2. **Exposure per tilt** (`EXPOSURE`) — e⁻/Å² per tilt image (e.g., `2.36`, `3.2`)
3. **Frame format + type** — cannot be derived from filenames alone; determine from file extension first, then frame count.
   - **Extension check (list the data dir):**
     - `*.eer` → Falcon 4 EER movie (always fractionated; needs frame grouping)
     - `*.tif` / `*.tiff` → TIFF movie stack (fractionated)
     - `*.mrc` → could be either — inspect Z dimension (see Header Tools below): Z > 1 = fractionated, Z == 1 = single frame
   - Single-frame MRC → symlink to `average/` + `fs_ctf`
   - Fractionated MRC/TIFF/EER → `fs_motion_and_ctf` (with format-specific flags)
   - File naming and MDOC `NumberOfFrames` are not reliable — don't trust them for detection

   **Header tools — pick whichever is available:**
   ```bash
   # Preferred (PyTOM env is already required by this skill):
   conda activate "$PYTOM_ENV" && headerPyTom <file.mrc>

   # Alternative (only if IMOD is on PATH):
   header <file.mrc>

   # Fallback (works in any conda env with mrcfile active, e.g. WARP_ENV):
   python -c "import mrcfile, sys; m = mrcfile.open(sys.argv[1], permissive=True); print('shape:', m.data.shape); m.close()" <file.mrc>
   ```
   The `header` (IMOD) command is **not** assumed available. Default to `headerPyTom` since `PYTOM_ENV` is already required by the skill; fall back to the mrcfile one-liner only if PyTOM is unavailable.

   **Script-side MRC metadata readers:**
   - `warp_tiltseries_setup.slurm` reads unbinned frame X/Y with Python `mrcfile` after activating `WARP_ENV` (`m.header.nx`, `m.header.ny`). If `mrcfile` is missing and `AUTO_INSTALL_MRCFILE=1`, the script installs it into `WARP_ENV` with `python -m pip install mrcfile` and re-checks the import.
   - `warp_update_tomo_dims.slurm`, `gen_tm_jobs_aretomo.slurm`, and `gen_sphere_mask.slurm` run in `PYTOM_ENV` and read MRC dimensions with `headerPyTom`; do not import OPUS-ET/`cryodrgn` in these scripts.
   - `gen_training_mask.slurm`, `train_opuset.slurm`, and `train_opuset_fixed.slurm` run in `OPUSET_ENV`; OPUS-ET training `ANGPIX` is inferred only from STAR `_rlnDetectorPixelSize`, and mask shape/radius is handled through `dsdsh create_mask` / OPUS-ET readers.
4. **Gain reference** — required for most fractionated movies (EER, TIFF, raw MRC):
   - Path to gain MRC/DM4 (`--gain <file>`); ask user if present
   - Orientation flags (`--gain_flip_x`, `--gain_flip_y`, `--gain_transpose`) — ask only if the user already knows them; otherwise omit and let WARP's default apply
   - Skip entirely for already-gain-corrected single-frame MRC
5. **EER frame grouping** (EER only) — how many raw EER frames per dose fraction, e.g. `--eer_ngroups 10`. Ask user; typical: 10–20 groups per tilt.
6. **Defects map** (optional) — `--defects <file.txt>` for bad-pixel masking; only if user has one.

### Tilt Series
7. **Approximate tomo dimensions** (`TOMO_DIMS`) — ask the user for `cols×rows×Z` in **unbinned** pixels before creating tilt-series settings (e.g. `3840x3712x2000`).
   - `cols × rows` = detector size (fixed, from camera). If the user does not know it, derive X/Y from one unbinned frame or averaged tilt image: set `SAMPLE_FRAME` in `warp_tiltseries_setup.slurm`, or leave `TOMO_DIM_X/Y` blank so the script reads the first MRC under `warp_frameseries/average`. The script reads `m.header.nx` / `m.header.ny` with Python `mrcfile` in `WARP_ENV`; if `mrcfile` is missing, the script can auto-install it into `WARP_ENV`.
   - `Z` = an **upper bound on sample thickness** in unbinned pixels — only used to size AreTomo's reconstruction slab. Phase 3.5 rewrites it from the actual `_ali.mrc` output, so a rough estimate is fine. Defaults: `2000` for thin lamellae / thin cryo-ET, `3000–4000` for cells / thick specimens.
8. **Binning factor** (`BINNING_FACTOR`) — for `ts_stack` export and AreTomo (e.g. `4`, `8`). AreTomo runs on the **binned** stack, so it receives binned dims:
   - `VOL_Z = TOMO_DIM_Z / BINNING_FACTOR` → AreTomo `-VolZ`; `500` binned voxels is a practical starting value for many datasets, then adjust if the tomogram needs more or less Z margin
   - `ALIGN_Z` → AreTomo `-AlignZ`; set this close to the measured sample thickness in **binned voxels**, and always keep `ALIGN_Z < VOL_Z`
   - `ALIGN_ANGPIX = ANGPIX × BINNING_FACTOR` (stack export pixel size)
   The unbinned `TOMO_DIMS` stays in `warp_tiltseries.settings`; only AreTomo's CLI sees the binned values (computed inside the script).
   For `ALIGN_Z`, do not rely on a fixed fraction of `VOL_Z` when accuracy matters. A practical approach is to run global-only alignment / a quick 3D reconstruction at large binning, inspect an XZ slice, measure the specimen thickness, then set `ALIGN_Z` near that thickness. `VOL_Z` should remain larger to provide final-reconstruction Z margin.

### Particle / Template Matching
9. **Particle diameter** (`DIAMETER`) — in Å (e.g., `320`)
10. **Box sizes** — there are **two distinct boxes**, sized at different pixel sizes:
    - **TM template / mask box** (used in Phase 6 by `gen_sphere_mask.slurm` and consumed by `gen_tm_jobs_aretomo.slurm`): sized at `ALIGN_ANGPIX`, since TM runs on the binned reconstructions. Rule: `box ≈ round_to_even(DIAMETER / 0.75 / ALIGN_ANGPIX)` (particle occupies ~75% of box). Example: 320 Å at `ALIGN_ANGPIX=13.48 Å` → 32 px.
    - **Subtomo export box** (`SUBTOMO_BOX_SIZE` in `warp_export_particles.slurm`, `warp_m_export.slurm`, OPUS-ET training): sized at `OUTPUT_ANGPIX`. Rule: `SUBTOMO_BOX_SIZE ≈ round_to_even(DIAMETER / 0.75 / OUTPUT_ANGPIX)` (particle occupies ~75% of box), with `DIAMETER / 0.5 / OUTPUT_ANGPIX` for high-res / strong CTF cases. Example: 320 Å at `OUTPUT_ANGPIX=2.5 Å` → 171 px → round to 176.

    Both should be a multiple of 8 (preferably 16) for FFT efficiency. Box too small → CTF aliasing; too large → wasted memory and slower training.
11. **Subtomogram export pixel sizes** — `OUTPUT_ANGPIX` in `warp_export_particles.slurm` must be chosen by the user/agent for OPUS-ET training and is often **binned** after template matching. Do not derive it from `warp_tiltseries.settings`. `COORDS_ANGPIX` is the pixel size of coordinates in the input STAR; after this skill's PyTom→WARP conversion it is usually the unbinned `ANGPIX`.
12. **Template generation / TM template** — **always ask the user** for the source template map (`INPUT_MRC`) and its pixel size (`MAP_ANGPIX`); never guess or default to `ribo.mrc`. The default workflow is to run `gen_template_from_mrc.slurm` before template matching, using `ANGPIX`, `BINNING_FACTOR`, and `TM_BOX_SIZE`, then set `TEMPLATE_MRC` to the generated output (usually under `templates/`). Only skip template generation if the user confirms they already have a PyTom-ready template at exactly `ALIGN_ANGPIX = ANGPIX × BINNING_FACTOR` and the intended TM box size.
13. **TM sphere mask** (`TM_MASK_MRC`) — absolute path to the mask paired with the template; produced by `gen_sphere_mask.slurm` and consumed by `gen_tm_jobs_aretomo.slurm`. Default: `$WORK_DIR/templates/${TM_LABEL}_mask.mrc`. The mask box is read from the template MRC dimensions, and the default sphere radius is derived in pixels as `round(DIAMETER / 2 / ALIGN_ANGPIX)`. Keep `MASK_SIGMA=1` unless the user asks for a softer edge; the safety condition is `MASK_RADIUS + MASK_SIGMA < TEMPLATE_DIM / 2`, so the soft edge fits inside the template box. **Both scripts must reference the same mask path** — propagate one variable into both.
14. **Extraction non-max suppression radius** (`MASK_RADIUS`, pixels) — in extraction script; **derive** with the same physical radius as `round(DIAMETER / 2 / ALIGN_ANGPIX)`. Two peaks closer than this distance are merged. Wrong value → either duplicate hits (too small) or missed adjacent particles (too large).
15. **Candidates per tomogram** (`NUM_CANDIDATES`) — top-N peaks extracted per TS in `extract_tm_candidates_parallel.slurm`. Over-pick (5000–10000 for crowded specimens, lower for sparse) and filter by score in the resulting STAR.
16. **Extraction reference/template name** (`TEMPLATE`) — with this skill's default job layout, `gen_tm_jobs_aretomo.slurm` creates one job directory per tilt series/tomogram, and `run_tm_sequential.slurm` runs `localization.py` inside that directory. PyTom therefore writes `scores_<template>.em` and `angles_<template>.em` directly in each job directory (e.g. `scores_ribo.em`, `angles_reference19.em`), not under `job_dir/<template>/`. **Always ask the user which template/reference suffix they want to extract**, since users commonly run multiple TMs against the same tomograms. Set `TEMPLATE` to that suffix (e.g. `TEMPLATE="ribo"`, `TEMPLATE="26s"`, or `TEMPLATE="reference19"`). Only leave empty if the user confirms a single reference per job; the helper will otherwise auto-pick the first matched pair and warn if ambiguous. The extractor can also read per-reference subdirs for externally organized PyTom jobs, but that is not produced by `gen_tm_jobs_aretomo.slurm`.
17. **TM label** (`TM_LABEL`) — one species/template namespace, e.g. `ribosome`, `26s`, `hsp`. For multiple species, run Phase 6 once per `TM_LABEL` and keep all intermediate files under `template_matching/$TM_LABEL/`. Do not share `particles/`, `star_files/`, or `warp_star/` across species; it becomes impossible to know which picks belong to which template. Prefer keeping `TM_LABEL`, the generated template basename, and the PyTom extraction `TEMPLATE` suffix consistent unless the user has an existing naming convention.

### OPUS-ET Training
18. **Latent dimension** (`ZDIM`) — default `8`; increase for more expressivity and more heterogeneous datasets
19. **Number of epochs** (`NUM_EPOCHS`) — default `40`
20. **Number of GPUs** — for SLURM (`--nproc_per_node`)
21. **Training decoder size** (`TEMPLATERES`) — OPUS-ET `--templateres`; the decoder output size should be ~ `SUBTOMO_BOX_SIZE × downfrac / 0.75` so the particle occupies ~75% of the decoder output. This is typically smaller than `SUBTOMO_BOX_SIZE` because the encoder downsamples.
22. **Training subtomogram directory** (`DATADIR`) — the subtomogram export directory (e.g. `$WORK_DIR/warp_tiltseries/subtomo`, or per-species like `subtomo_ribo`). The training scripts pass `--datadir $(dirname "$DATADIR")` to OPUS-ET because the STAR file's `_rlnImageName` entries already include the subdirectory prefix. For mask creation, `$DATADIR` is used directly to find a sample subtomogram.
23. **Training mask** (`TRAINING_MASK_MRC`) — required by OPUS-ET training. Default: `$WORK_DIR/templates/${TM_LABEL}_training_mask.mrc`. For template-matching filtering, a broad centered sphere mask is acceptable: run `gen_training_mask.slurm` with `MODE="sphere"` (the default, soft edge 2 voxels, 3 dilate iterations), which samples an exported subtomogram from `DATADIR` for shape/header (so the output mask matches `SUBTOMO_BOX_SIZE` at `OUTPUT_ANGPIX`). The default sphere diameter is 85% of the box (`MASK_RADIUS_FRACTION=0.85`). Use `MODE="density"` only when a clean consensus map should define a tighter molecular mask. If the mask is missing at training time and `AUTO_CREATE_TRAINING_MASK=1`, `train_opuset.slurm` and `train_opuset_fixed.slurm` can still create a fallback spherical mask from an exported subtomogram using `dsdsh create_mask --sphere-radius`.
24. **Training geometry overrides** (`ANGPIX`, `TILT_RANGE`, `TILT_STEP`) — in `train_opuset.slurm` and `train_opuset_fixed.slurm`, leave `ANGPIX` blank to auto-detect only from `_rlnDetectorPixelSize` in the input particle STAR file. Do not infer training `ANGPIX` from `OUTPUT_ANGPIX` in `warp_export_particles.slurm` or from `warp_tiltseries.settings`. Tilt range/step are read from the first `.tlt` file under `TILTSTACK_DIR`, normally `warp_tiltseries/tiltstack`. Set them manually only when the auto-detected values are wrong or the dataset uses non-standard metadata.
25. **Fixed-mode subset split** — Phase 8c needs two independent particle halves for half-map reconstruction. The input is a **selected** subset of particles (e.g., from OPUS-ET analysis picking a specific conformational state), not a blind split of the full matching STAR. Once the user has a selected STAR (`sel.star`), split it with:
    ```bash
    dsdsh convert_star sel.star --subset-label 1 --angpix <ANGPIX> -o sel_subset1.star
    dsdsh convert_star sel.star --subset-label 2 --angpix <ANGPIX> -o sel_subset2.star
    ```
    Run 8c twice, once per subset, with `FIXED_SUBSET_LABEL=1` and `FIXED_SUBSET_LABEL=2`.

### Environment Paths
**These are required for ALL phases — collect them first before anything else.**
25. **Project directory** (`WORK_DIR`) — absolute path where data and scripts will run
26. **WARP install path** (`WARP_DIR`) — absolute path to `publish/` dir, e.g. `/home/user/warp/Release/linux-x64/publish`. **Must be the alncat fork on the `alncat` branch** (<https://github.com/alncat/warp>), not upstream <https://github.com/warpem/warp>: this skill's `ts_export_particles` calls rely on `--dont_correct_ctf_3d` and `--output_ctf_csv`, both of which exist only in the fork (OPUS-ET ingests the per-particle CTF CSV and applies CTF correction internally). Verify with `WarpTools ts_export_particles --help | grep dont_correct_ctf_3d` — empty output means stock WARP. To install the fork:
    ```bash
    git clone https://github.com/alncat/warp.git
    cd warp
    git checkout alncat
    # Then follow the upstream WARP build instructions to produce
    # Release/linux-x64/publish/. Point WARP_DIR at that publish/ directory.
    ```
27. **WARP conda env** (`WARP_ENV`) — e.g. `warp_build`
28. **OPUS-ET conda env** (`OPUSET_ENV`) — e.g. `opuset_env`
29. **PyTOM conda env** (`PYTOM_ENV`) — e.g. `pytom_env` (template matching phases)
30. **conda lib path** (`CONDA_LIB`) — `lib/` dir of WARP conda env; derive with:
    ```bash
    conda activate <WARP_ENV> && echo $CONDA_PREFIX/lib
    ```
    e.g. `/home/user/.conda/envs/warp_build/lib`
31. **AreTomo cluster modules** (`MODULE_CUDA`, `MODULE_COMPILER`) — only `warp_aretomo_align_negate.slurm` uses `module load`; set these only if AreTomo2 requires cluster modules on the user's system, otherwise set them to `""`.
32. **SLURM partitions and GPU count** — ask once, propagate into every script:
    - `CPU_PARTITION` — partition name for CPU-only jobs (scripts shipped as `--partition=normal`). Examples: `normal`, `compute`, `cpu`.
    - `GPU_PARTITION` — partition name for GPU jobs (scripts shipped as `--partition=gpu` or `--partition=normal`). Examples: `gpu`, `gpu-a100`, `volta`.
    - `GRES_GPU` — full `--gres` value, e.g. `gpu:4`, `gpu:a100:4`, `gpu:1`. Some sites require the GPU type (`gpu:a100:4`); check `sinfo -o "%G"`.
    - `NUM_GPUS` — integer, must equal the count in `GRES_GPU`. Used by the training scripts (`--nproc_per_node`, `--num-gpus`).
    These default values in the scripts assume a generic SLURM setup — **do not submit without confirming** the partition names exist on the user's cluster (`sinfo -s` lists available partitions).

### Optional
33. **Output directory prefix** (`OUTPUT_DIR`) — e.g., `opuset/${TM_LABEL}/z${ZDIM}`

### M Refinement
34. **Half-maps** (`M_HALF1`, `M_HALF2`) — after Phase 8d, these default to `opuset/<TM_LABEL>/half1.mrc` and `half2.mrc`. User can also provide externally generated half-maps if skipping Phase 8c/8d.
35. **Resample pixel size** (`ANGPIX_RESAMPLE`) — MCore working pixel size for species creation. Defaults to `$ANGPIX` (unbinned detector pixel size). Set to a coarser value for faster refinement with lower memory usage.
36. **Half-map mask** (`M_MASK`) — defaults to `opuset/<TM_LABEL>/halfmap_mask.mrc` from Phase 8d. User can provide a custom mask.
37. **Half-map mask threshold** (`M_HALFMAP_THRESHOLD`) — density threshold for creating the half-map mask in Phase 8d. Only positive density above this value is kept. User must provide this.
38. **RELION particles STAR** (`M_PARTICLES_STAR`) — RELION-format particle STAR file for MTools create_species. User must provide this.
39. **M mask threshold** (`M_MASK_THRESHOLD`) — density threshold for MTools update_mask between MCore passes (optional, only needed if running Md).

---

## Quick Start

1. **Copy examples:** `cp pipeline.example.conf pipeline.conf && cp species.example.conf species.conf`
2. **Edit `pipeline.conf` and `species.conf`** — fill all variables (paths, pixel sizes, labels, etc.)
3. **Run `bash validate.sh --phase 1`** — checks placeholders, paths, tools, derived sanity
4. **Submit Phase 1:** `sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_frameseries_import.slurm`

### Variables Reference

All variables are defined in `pipeline.example.conf` and `species.example.conf` — those files are the canonical reference. Copy them to `.conf`, edit, then run `validate.sh` to check them. Derived values (`ALIGN_ANGPIX`, directory paths, etc.) are computed automatically when the configs are sourced.

### Configuration (pipeline.conf + species.conf)

**Two config files** — copy from examples, then edit:

```bash
cp pipeline.example.conf pipeline.conf
cp species.example.conf species.conf
vim pipeline.conf    # tilt-series-wide: paths, microscope, detector, cluster, binning
vim species.conf     # per-species: TM_LABEL, DIAMETER, box sizes, training params
```

`pipeline.conf` is shared by all species. `species.conf` is per-species — keep separate copies for different species (e.g. `species_ribo.conf`, `species_26s.conf`). Override via `SPECIES_CONF=species_26s.conf sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/...`. 

**SLURM copies scripts to a temp directory, so `$0` is not the original path.** Always submit with `SKILL_DIR` so scripts can find their config files:

```bash
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_frameseries_import.slurm
```

In scripts, `SCRIPT_DIR` falls back to `$0` if `SKILL_DIR` is unset:

```bash
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")" && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found..."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found..."; exit 1; }
```

### Pre-flight Validation (validate.sh)

**Run before submitting any job:**
```bash
bash validate.sh                        # full check
bash validate.sh --phase 1              # Phase 1 only
bash validate.sh --phase 6 --species species_ribo.conf  # Phase 6, specific species
bash validate.sh --phase 7 --dry-run    # resolve paths + print commands
```

All pre-submission checks are handled by `validate.sh`:
- Unresolved placeholders in `pipeline.conf`, `species.conf`, and all SLURM scripts
- Environment paths (`WORK_DIR`, `WarpTools`, `CONDA_LIB`)
- Conda environments (`WARP_ENV`, `OPUSET_ENV`, `PYTOM_ENV`)
- Required tools (AreTomo2, dsdsh, etc.) — phase-aware
- Derived value sanity (CTF range ≥ 2×ANGPIX, box covers particle, mask fits template)
- Per-tomostar completion checks for the current phase
- SLURM script syntax (`bash -n`)
- `--dry-run` prints resolved paths and commands without submitting

### WARP Environment Notes

> ⚠ **`ulimit -v unlimited` is mandatory.** WARP, MTools, MCore, and AreTomo allocate very large virtual address spaces. Without it they fail at startup with cryptic errors. Every shipped SLURM script already includes this line.

**`WARP_FORCE_MRC_FLOAT32=1`** forces 32-bit float MRC outputs (set in every shipped script). Disk-space tradeoff: 32-bit floats ~2× the size of WARP's compact int16 default. Drop the export if downstream tools handle compressed format and disk is tight.

### Full Pipeline (SLURM)
```bash
# All sbatch commands must pass SKILL_DIR so scripts find pipeline.conf:
#   sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/...

# Phase 1: Frame series
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_frameseries_import.slurm

# Phase 2: Tilt series setup
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_tiltseries_setup.slurm

# Phase 3: Export stacks + AreTomo2 alignment
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_export_stacks.slurm
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_aretomo_align_negate.slurm

# Phase 3.5: REQUIRED — update TOMO_DIMS from AreTomo output
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_update_tomo_dims.slurm

# Phase 4: Import alignments
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_import_alignments.slurm

# Phase 5: CTF + reconstruction
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_ts_ctf.slurm
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_ts_reconstruct.slurm

# 💡 Verify Phase 3–5 alignment chain: AreTomo vs WARP reconstruction dimensions
#    Both are at ALIGN_ANGPIX, so nx × ny × nz must match exactly:
#
#      TS=TS_026
#      headerPyTom warp_tiltseries/tiltstack/$TS/${TS}_ali.mrc | grep 'Number of columns'
#      headerPyTom warp_tiltseries/reconstruction/${TS}_*Apx.mrc | grep 'Number of columns'
#
#    Mismatch → Phase 3.5 skipped, or --alignment_angpix wrong in Phase 4,
#    or TOMO_DIMS X/Y incorrect. Fix the source, re-run from Phase 3.5/4.
#    Details: references/phases.md § sanity check.

# Phase 6: Template matching
# All shared variables are already in pipeline.conf + species.conf.
# Scripts source both automatically — no per-script CONFIG editing needed.
# Agent MUST fill these in species.conf before submitting:
#   TM_LABEL, DIAMETER, TM_BOX_SIZE, SUBTOMO_BOX_SIZE
#   INPUT_MRC, MAP_ANGPIX (template source)
#   NUM_CANDIDATES, MASK_RADIUS, TEMPLATE (extraction)
# Agent MUST fill these in pipeline.conf:
#   ANGPIX, BINNING_FACTOR (→ ALIGN_ANGPIX derived automatically)
# Script-specific overrides (edit only if defaults don't fit):
#   gen_tm_jobs_aretomo.slurm: SEARCH_REGION=""  (full X/Y, z_start=20)
#   extract_tm_candidates_parallel.slurm: MASK_RADIUS=14  (non-max suppression)
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/gen_template_from_mrc.slurm
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/gen_sphere_mask.slurm
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/gen_tm_jobs_aretomo.slurm
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/run_tm_sequential.slurm
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/extract_tm_candidates_parallel.slurm
# PyTom XML → per-tilt-series STAR. Canonical path: two SLURM steps below.
# Do not use a combined particles.star here.
# convert_to_star.slurm and convert_pytom_to_warp.slurm must use the same
# TM_LABEL, so they read/write template_matching/$TM_LABEL/star_files and warp_star.
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/convert_to_star.slurm
# convert_to_star.slurm sources pipeline.conf for ANGPIX (unbinned) and BINNING_FACTOR.
# These map to convert.py's --pixelSize and --binPyTom. Verify both are set in pipeline.conf.
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/convert_pytom_to_warp.slurm

# Phase 7: Export subtomograms
# All variables are in pipeline.conf + species.conf — no per-script editing needed.
# Verify in species.conf: TM_LABEL, SUBTOMO_BOX_SIZE, DIAMETER, OUTPUT_ANGPIX
# Verify in pipeline.conf: COORDS_ANGPIX (= ANGPIX)
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/warp_export_particles.slurm

# Phase 8: OPUS-ET training (sub-phases: 8a mask → 8b grad → 8c fixed)
# All variables are in pipeline.conf + species.conf — no per-script editing needed.
# Verify in species.conf: TM_LABEL, ZDIM, NUM_EPOCHS, BATCH_SIZE,
#   SUBTOMO_BOX_SIZE, TEMPLATERES, MASK_SOFT_EDGE, MASK_RADIUS_FRACTION
# Verify in pipeline.conf: NUM_GPUS
# For fixed-mode (Phase 8c): run dsdsh convert_star --subset-label 1 on the Phase 7 STAR first.
# Phase 8b auto-creates a fallback sphere mask if TRAINING_MASK_MRC is missing.
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/gen_training_mask.slurm  # Phase 8a — optional explicit mask
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/train_opuset.slurm        # Phase 8b — heterogeneity
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/train_opuset_fixed.slurm   # Phase 8c — fixed half-maps
sbatch --export=ALL,SKILL_DIR="$(pwd)" scripts/prepare_m_halfmaps.slurm    # Phase 8d — link half-maps + mask
```

**For detailed per-phase commands:** read `references/phases.md`

---

## Phase Summary

| Phase | What happens | Key script(s) |
|-------|-------------|---------------|
| 1: Frame series | Settings → link/motion correct → CTF | `warp_frameseries_import.slurm` |
| 2: Tilt series | Import from MDOC → create settings | `warp_tiltseries_setup.slurm` |
| 3: Alignment | Export stacks → AreTomo2 → negate angles | `warp_export_stacks.slurm`, `warp_aretomo_align_negate.slurm` |
| 3.5: Update dims | Read binned dims from `_ali.mrc` → UNBINNED `TOMO_DIMS` → re-run `create_settings` | `warp_update_tomo_dims.slurm` |
| 4: Import alignments | Import .xf/.tlt into WARP | `warp_import_alignments.slurm` |
| 5: CTF + recon | Defocus handedness → CTF → reconstruct | `warp_ts_ctf.slurm`, `warp_ts_reconstruct.slurm` |
| 6: Template matching | Generate TM-ready template → mask → jobs → run TM → extract → convert | `gen_template_from_mrc.slurm`, `gen_sphere_mask.slurm`, `gen_tm_jobs_aretomo.slurm`, `run_tm_sequential.slurm`, `extract_tm_candidates_parallel.slurm`, `convert_to_star.slurm`, `convert_pytom_to_warp.slurm` |
| 7: Export | Export subtomograms for OPUS-ET | `warp_export_particles.slurm` |
| 8a: Training mask | Generate sphere/density mask at `OUTPUT_ANGPIX` from subtomo sample | `gen_training_mask.slurm` |
| 8b: Train (grad) | Train OPUS-ET heterogeneity model | `train_opuset.slurm` |
| 8c: Train (fixed) | Fixed-mode averaging → half-maps for M refinement | `train_opuset_fixed.slurm` |
| 8d: Prep half-maps | Link half-maps + create mask for M refinement | `prepare_m_halfmaps.slurm` |
| Ma: M setup | create_population + create_source (one-time) | `warp_m_setup.slurm` |
| Mb: M species | create_species with half-maps + mask + particles | `warp_m_create_species.slurm` |
| Mc: M refine | MCore iterative refinement (re-submit each pass) | `warp_m_refine.slurm` |
| Md: M mask | update_mask between MCore passes | `warp_m_update_mask.slurm` |
| Me: M export | Re-export refined particles | `warp_m_export.slurm` |

---

## SLURM Scripts Reference

| Script | Purpose | Memory |
|--------|---------|--------|
| `warp_frameseries_import.slurm` | Frame series settings + link + CTF | 128 GB |
| `warp_tiltseries_setup.slurm` | Tilt series settings + MDOC import | 128 GB |
| `warp_export_stacks.slurm` | Export tilt stacks for alignment | 64 GB |
| `warp_aretomo_align_negate.slurm` | AreTomo2 with angle negation | 32 GB |
| `warp_update_tomo_dims.slurm` | Re-run `create_settings` with UNBINNED dims from `_ali.mrc` (Phase 3.5) | 16 GB |
| `warp_import_alignments.slurm` | Import .xf/.tlt into WARP | 64 GB |
| `warp_ts_ctf.slurm` | Defocus handedness check + CTF | 256 GB |
| `warp_ts_reconstruct.slurm` | Tomogram reconstruction | 256 GB |
| `gen_sphere_mask.slurm` | Sphere mask for TM | 16 GB |
| `gen_template_from_mrc.slurm` | Generate PyTom TM template from user MRC map | 32 GB |
| `gen_tm_jobs_aretomo.slurm` | Generate PyTOM job XMLs | 16 GB |
| `run_tm_sequential.slurm` | Run TM jobs sequentially | 256 GB |
| `extract_tm_candidates_parallel.slurm` | Extract particles (parallel) | 256 GB |
| `convert_to_star.slurm` | PyTOM XML → per-tilt-series STAR (canonical; wraps PyTom `convert.py`) | 64 GB |
| `convert_pytom_to_warp.slurm` | PyTOM STAR → WARP format | 32 GB |
| `warp_export_particles.slurm` | Export subtomograms | 128 GB |
| `gen_training_mask.slurm` | Training mask | 16 GB |
| `train_opuset.slurm` | Train OPUS-ET heterogeneity model (4 GPUs) | 256 GB |
| `train_opuset_fixed.slurm` | Train OPUS-ET fixed-mode (encoder-free, homogeneous average; run twice on even/odd splits to produce M half-maps) | 256 GB |
| `warp_m_setup.slurm` | M refinement: create_population + create_source (one-shot) — `references/advanced.md` | 16 GB |
| `warp_m_create_species.slurm` | M refinement: create_species (per species) | 64 GB |
| `warp_m_update_mask.slurm` | M refinement: update_mask (between passes only — needs a completed refine first) | 32 GB |
| `warp_m_refine.slurm` | M refinement: MCore pass (iterative — re-submit each iteration) | 256 GB |
| `warp_m_export.slurm` | M refinement: re-export refined particles for one species at a time via ts_export_particles | 128 GB |

For `warp_m_update_mask.slurm` and `warp_m_export.slurm`, `SPECIES_BASE` may be either the original species base name or the full hashed folder name. If `m/species/<species>_*` matches multiple folders, use the full folder name, e.g. `SPECIES_BASE="ribosome_1199b7f2"`, so one species is selected unambiguously.

**All scripts** source `pipeline.conf` (and `species.conf` for Phase 6+). Fill the config files — not individual scripts. Run `validate.sh` before submitting to catch any remaining placeholders or path issues.

**Cluster-specific SLURM directives (also do this for every script).** Scripts
ship with generic `--partition`, `--gres`, and `--time` defaults. Before the
first submission on a new cluster:

- For scripts with no `--gres=gpu:*`, replace `--partition=normal` with `--partition=$CPU_PARTITION`. This includes CPU-only scripts such as `convert_*`, `gen_*`, `extract_*`, `warp_tiltseries_setup`, `warp_m_setup`, `warp_m_create_species`, `warp_m_update_mask`, `warp_update_tomo_dims`, and `warp_import_alignments`.
- For any script that has `--gres=gpu:*`, use `--partition=$GPU_PARTITION` even if the shipped partition is `normal`. This includes `warp_frameseries_import.slurm` and `warp_export_stacks.slurm`.
- Replace the shipped GPU request (`--gres=gpu:1`, `--gres=gpu:4`, etc.) with the cluster-specific GPU request and adjust `--nproc_per_node` / `--num-gpus` in the training scripts to match `$NUM_GPUS`.
- `--time` defaults are guesses (12h CTF, 24h recon, 48h TM, 120h training). If the user knows their cluster's max time or expected scale (e.g. 200 TS vs 20), bump or lower accordingly.

If `sinfo -s` lists no partition called `normal` or `gpu`, **do not submit** with the shipped defaults — SLURM will reject the job immediately.

**Placeholders are caught by `validate.sh`** (see Pre-flight Validation above). It scans `pipeline.conf`, `species.conf`, and all SLURM scripts for `<placeholder>` tokens and `/path/to/` defaults.

---

## Deployment Files

| File | Purpose |
|------|---------|
| `pipeline.example.conf` | Template — copy to `pipeline.conf` and edit |
| `species.example.conf` | Template — copy to `species.conf` and edit (one per species) |
| `scripts/manifest.yml` | Machine-readable phase map — scripts, vars, tools, outputs per phase |
| `validate.sh` | Phase-aware pre-flight check, species-selectable |

**Setup:** `cp pipeline.example.conf pipeline.conf && cp species.example.conf species.conf`, then edit both. Scripts exit with a clear error if the real `.conf` file is missing.

## File Organization

Skill deployment files (shipped with the skill):
```
scripts/                        # All SLURM scripts + manifest.yml + extract_candidates_parallel.py
validate.sh                     # Phase-aware pre-flight check
pipeline.example.conf           # Copy to pipeline.conf and edit
species.example.conf            # Copy to species.conf and edit (one per species)
pipeline_flowchart.html         # Visual reference
```

Pipeline runtime output under WORK_DIR:
```
WORK_DIR/
├── logs/                       # SLURM .out / .err files
├── mdoc/                      # MDOC files
├── tomostar/                  # Canonical tilt series list (*.tomostar)
├── warp_frameseries/          # Phase 1
│   └── average/               # Linked or motion-corrected tilt images
├── warp_tiltseries/           # Phases 2–5
│   ├── tiltstack/TS_XXX/      # Exported stacks + AreTomo output
│   │   ├── TS_XXX.st          # Tilt stack
│   │   ├── TS_XXX.rawtlt      # Raw angles
│   │   ├── TS_XXX_neg.rawtlt  # Negated (for AreTomo)
│   │   ├── TS_XXX_ali.mrc     # AreTomo tomogram
│   │   ├── TS_XXX.xf          # Transforms (for WARP)
│   │   └── TS_XXX.tlt         # Refined angles (for WARP)
│   ├── reconstruction/        # CTF-corrected tomograms (TS_XXX_<apix>Apx.mrc)
│   ├── subtomo/               # Phase 7: exported subtomograms for OPUS-ET
│   │   └── TS_XXX/*.mrc        #   per-tilt-series subtomograms (at OUTPUT_ANGPIX)
│   ├── <source>.source        # MTools source file from warp_m_setup.slurm
│   └── <species>_<hash>_matching_refined.star  # Optional M export STAR, one per species
├── templates/                  # Template generation + masks (Phase 6/8)
│   ├── <TM_LABEL>_tm.mrc        #   template at ALIGN_ANGPIX (gen_template_from_mrc)
│   ├── <TM_LABEL>_mask.mrc      #   TM sphere mask (gen_sphere_mask)
│   └── <TM_LABEL>_training_mask.mrc  # training mask (gen_training_mask)
├── template_matching/
│   └── <TM_LABEL>/            # One namespace per species/template
│       ├── jobs/              # PyTOM job XMLs + scores/angles
│       ├── particles/         # Extracted particle XMLs
│       ├── star_files/        # Per-tilt-series PyTom STAR files
│       └── warp_star/         # WARP-compatible STAR files
├── m/                         # Optional MTools refinement workspace
│   ├── <name>.population      # Created by warp_m_setup.slurm
│   └── species/
│       └── <species>_<hash>/  # Created by warp_m_create_species.slurm
│           ├── <species>.species
│           ├── *_particles.star
│           ├── *_relion.star  # Produced by warp_m_export.slurm before export
│           ├── <species>_half1.mrc
│           └── <species>_half2.mrc
└── opuset/                     # Phase 8: OPUS-ET training output
    └── <TM_LABEL>/               #   namespaced by species
        ├── z12/                  #   heterogeneity training (ZDIM=12)
        │   ├── weights.*.pkl
        │   ├── z.*.pkl
        │   └── config.pkl
        └── fixed_subset<N>/      #   fixed-mode half-map reconstruction
            ├── weights.*.pkl
            └── config.pkl
```

---

## Critical Issues (Quick Reference)

- **CTF range max > Nyquist**: reduce `--range_max`; must be ≥ 2× angpix
- **GLIBCXX error**: missing `LD_LIBRARY_PATH` — set `${CONDA_LIB}:${WARP_DIR}`
- **OutOfMemory**: use 256 GB+, set `ulimit -v unlimited`, reduce `--perdevice` to 1
- **Wrong tilt convention**: WARP writes angles from positive max down to negative min (e.g. +54, +51, …, -54); AreTomo expects negative max up to positive max (e.g. -54, -51, …, +54). Fix: `awk '{print -$1}'` negates each angle — order is preserved in the respective sign convention.
- **WARP tomo_dimensions must be UNBINNED**: AreTomo output is binned — multiply by binning factor. Phase 2's initial `TOMO_DIMS` is only a guess; run Phase 3.5 (`warp_update_tomo_dims.slurm`) after AreTomo and before Phase 4 to rewrite settings with the true unbinned size. Skipping this silently corrupts coordinates in Phases 5–7.
- **Verify after Phase 5 — AreTomo vs WARP reconstruction must agree**: tell the user to compare `headerPyTom warp_tiltseries/tiltstack/$TS/${TS}_ali.mrc` against `headerPyTom warp_tiltseries/reconstruction/${TS}_*Apx.mrc`. WARP names tomograms with a pixel-size suffix (e.g. `TS_026_7.84Apx.mrc`). Both are at `ALIGN_ANGPIX`, so `nx × ny × nz` must match exactly. A mismatch indicates Phase 3.5 was skipped, `--alignment_angpix` was wrong in Phase 4, or `TOMO_DIMS` X/Y was incorrect. See `references/phases.md` for the failure-mode breakdown.
- **--split must be absolute path** in OPUS-ET training, placed inside the output directory
- **Tomostar loop only**: always iterate from `tomostar/*.tomostar`, not from MRC glob

**For detailed troubleshooting and script patterns:** read `references/gotchas.md`

**For MTools refinement and OPUS-ET averaging:** read `references/advanced.md`

**For step-by-step phase commands:** read `references/phases.md`

**External reference for WarpTools API:**
<https://warpem.github.io/warp/reference/warptools/api/>

**External reference for MTools API:**
<https://warpem.github.io/warp/reference/mtools/api/>

**External reference for PyTom template matching:**
<https://github.com/SBC-Utrecht/PyTom/wiki/Template-matching>

**External reference for AreTomo `-VolZ` / `-AlignZ`:**
<https://gensoft.pasteur.fr/docs/AreTomo/1.3.4/AreTomoManual_1.3.0_09292022.pdf>
