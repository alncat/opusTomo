# Advanced Workflows

---

## WARP Refinement with MTools

After exporting particles from WARP, use MTools for iterative refinement.

### Prerequisites: half maps from OPUS-ET fixed mode

`create_species` needs input half maps (`half1.mrc` and `half2.mrc`, or any
user-supplied paths). After M refinement, MCore writes species-prefixed half
maps such as `<species>_half1.mrc` and `<species>_half2.mrc`. The skill can
produce initial half maps by
training **two OPUS-ET fixed-mode runs**, one on each half-set of the Phase 7
particles:

```bash
# 1. Split the Phase 7 STAR into the two halves (rlnRandomSubset = 1, 2)
dsdsh convert_star warp_tiltseries/matchingribo.star $ANGPIX --subset-label 1
dsdsh convert_star warp_tiltseries/matchingribo.star $ANGPIX --subset-label 2
# → matchingribo_subset1.star, matchingribo_subset2.star

# 2. Train fixed-mode (encoder-free, single homogeneous average) on each half.
#    See "OPUS-ET Subtomogram Averaging (Fixed Latent Code)" below for the
#    exact command. Run it twice — once per subset STAR, with different -o.
#    The reconstructed volume from each run is your half-map.
```

M accepts half-maps at any pixel size — `create_species --angpix_resample`
handles the resampling on import, so you can pass the OPUS-ET volumes
directly without pre-resampling.

**Mask:** M requires a *thresholded* mask shaped to the molecule, not a sphere
mask — `create_species` rejects loose sphere masks because they let too much
solvent noise into the FSC calculation. Generate one by thresholding a
half-map (or the consensus average) at a contour where the protein density is
cleanly separated from solvent, then dilate slightly with a soft edge:

```bash
# Pick <threshold> by inspecting the half-map in ChimeraX / 3dmod
# (same procedure as warp_m_update_mask.slurm). Then build the mask, e.g.:
relion_mask_create \
    --i half1.mrc \
    --o mask.mrc \
    --ini_threshold <threshold> \
    --extend_inimask 2 \
    --width_soft_edge 4
```

`gen_sphere_mask.slurm` / `gen_training_mask.slurm` produce sphere masks for
template matching and for the OPUS-ET training loss; **don't reuse them as
M's `--mask`**.

### 1. Create Population

**Use a single population per project, even with multiple species.** A
population is tied to a *data source* (the tilt-series + CTF + alignments), not
to a particular molecule. When the user has run multi-template TM (ribosome,
26S, HSP, …) against the same tilt series, all of those species belong in
**one** population — they share the same image-warp grid, per-tilt CTF, and
stage geometry, and M jointly refines those shared parameters across species in
each pass. Creating one population per template wastes compute and prevents
joint refinement of the shared imaging model.

Add each molecule with a separate `warp_m_create_species.slurm` invocation
(re-run the script with a different `SPECIES_BASE` and matching half-maps /
mask / particles STAR), pointing every invocation at the **same**
`POPULATION_NAME`.

```bash
MTools create_population \
    --directory m \
    --name <population_name>
# → m/<name>.population
```

### 2. Create Data Source
```bash
MTools create_source \
    --name <source_name> \
    --population m/<population>.population \
    --processing_settings warp_tiltseries.settings
```

### 3. Create Species
```bash
MTools create_species \
    --population m/<population>.population \
    --name <species_name> \
    --diameter <diameter_Å> \
    --sym <C1|C2|D2|...> \
    --temporal_samples 1 \
    --half1 <half1.mrc> \
    --half2 <half2.mrc> \
    --mask <mask.mrc> \
    --particles_relion <particles.star> \
    --angpix_resample <unbinned_angpix> \
    --lowpass <lowpass_Å>
```

**Key parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--angpix_resample` | Target pixel size (usually unbinned) | `2.132` |
| `--lowpass` | Lowpass filter in Å | `15` |
| `--sym` | Symmetry group | `C1`, `C2`, `D2` |

**Species naming convention:** WARP appends a hash suffix to the folder:
```bash
# Auto-find species folder:
SPECIES_DIR=$(find m/species -maxdepth 1 -type d -name "${SPECIES_BASE}_*" | head -1)
SPECIES_NAME=$(basename "$SPECIES_DIR")   # e.g., nucleosome_1199b7f2
```

If the same base name has multiple hashed species folders, do not let a script
pick one. Set `SPECIES_BASE` to the full hashed folder name instead:

```bash
SPECIES_BASE=nucleosome_1199b7f2
```

### 4. Run Refinement (MCore)
```bash
MCore \
    --population m/<population>.population \
    --refine_imagewarp 4x4 \
    --refine_particles \
    --ctf_defocus \
    --perdevice_refine 1
```

**Options:**
| Flag | Description |
|------|-------------|
| `--refine_imagewarp 4x4` | Refine image deformation with 4×4 patch grid |
| `--refine_particles` | Refine particle poses (rotations + translations) |
| `--ctf_defocus` | Refine CTF defocus per tilt |
| `--perdevice_refine 1` | Threads per GPU |

**Output:** refined half maps in `m/species/<name>/<species>_half1.mrc`,
`<species>_half2.mrc`

### 5. Update Mask (between refinement passes)

Pick `--threshold` by **inspecting `<SPECIES_BASE>_filt.mrc` in ChimeraX or
`3dmod`**: lower the contour slider until the isosurface tightly wraps the
protein density and excludes solvent — that contour value is the threshold.
There is no universal default; it depends on map scaling and resolution.

```bash
SPECIES_DIR=$(find "$WORK_DIR/m/species" -maxdepth 1 -type d -name "${SPECIES_BASE}_*" | head -1)
SPECIES_NAME=$(basename "$SPECIES_DIR")

MTools update_mask \
    --population m/<population>.population \
    --species "${SPECIES_DIR}/${SPECIES_BASE}.species" \
    --map "${SPECIES_DIR}/${SPECIES_BASE}_filt.mrc" \
    --threshold <threshold> \
    --dilate 2
# Then re-run refinement
```

### 6. Export Refined Particles

MCore writes `<species>_particles.star` in WARP-flavored STAR format, but
`WarpTools ts_export_particles` consumes the **RELION** dialect. The export
is therefore a two-step pipeline:

```bash
SPECIES_DIR=$(find "$WORK_DIR/m/species" -maxdepth 1 -type d -name "${SPECIES_BASE}_*" | head -1)

# Step 1: convert WARP-style particles.star → RELION-style relion.star
for ps in "${SPECIES_DIR}"/*_particles.star; do
    dsdsh convert_warp "$ps" -o "${ps%_particles.star}_relion.star"
done

# Step 2: re-export subtomos using the converted STAR
WarpTools ts_export_particles \
    --settings warp_tiltseries.settings \
    --input_directory "$SPECIES_DIR" \
    --input_pattern "*_relion.star" \
    --output_star "warp_tiltseries/${SPECIES_NAME}_matching_refined.star" \
    --output_angpix $OUTPUT_ANGPIX \
    --box $SUBTOMO_BOX_SIZE \
    --diameter $DIAMETER \
    --relative_output_paths \
    --3d \
    --coords_angpix $COORDS_ANGPIX \
    --output_ctf_csv \
    --dont_correct_ctf_3d
```

`warp_m_export.slurm` runs both steps for **one species at a time**. Do not
combine species in one export; keep `SPECIES_BASE` explicit and export each
species separately so STAR files and subtomogram paths remain interpretable.
`dsdsh convert_warp` lives in
`DSDSH_ENV`, so the script swaps conda envs around the conversion call.

**Key:** `--coords_angpix` is the original pixel size from refinement
(unbinned); `--output_angpix` is the export target.

The same conversion is also useful when you want the refined poses only
(without re-exporting subtomos) — for example, to feed refined metadata into
RELION or another downstream tool.

### Complete refinement workflow

Five SLURM wrappers, mapped to the natural cardinality of each step:

| Script | Cardinality | Purpose |
|---|---|---|
| `warp_export_particles.slurm` | once | export the initial particles (canonical Phase 7) |
| `warp_m_setup.slurm` | once per project | `create_population` + `create_source` |
| `warp_m_create_species.slurm` | once per species | `create_species` (re-run with a different `SPECIES_BASE` to add another species to the same population) |
| `warp_m_update_mask.slurm` | between passes only | `update_mask` — must run AFTER at least one refinement pass (consumes `<species>_filt.mrc`) and BEFORE the next pass; no point after the final pass |
| `warp_m_refine.slurm` | iterative | one MCore pass — re-submit each iteration |
| `warp_m_export.slurm` | once per species after convergence | `dsdsh convert_warp` on that species' `*_particles.star`, then `ts_export_particles` on the resulting `*_relion.star`; writes a species-specific output STAR |

Typical order (with iteration):

```bash
sbatch scripts/warp_export_particles.slurm    # initial particles
sbatch scripts/warp_m_setup.slurm             # population + source (one-shot)
sbatch scripts/warp_m_create_species.slurm    # add species A (repeat for B, C…)
sbatch scripts/warp_m_refine.slurm            # pass 1 (mask = the one from create_species)
sbatch scripts/warp_m_update_mask.slurm       # optional: re-tighten mask using pass-1 map
sbatch scripts/warp_m_refine.slurm            # pass 2 with the updated mask
# (repeat refine ± update_mask until converged)
sbatch scripts/warp_m_export.slurm            # re-export one converged species; repeat per species
```

**Mask cadence.** The first refinement pass uses the mask supplied to
`warp_m_create_species.slurm`. `update_mask` reads the filtered half-map
(`<species>_filt.mrc`) produced by a refinement pass, so it can only run
*after* a pass has completed and only matters *before* a subsequent pass.

Scripts auto-resolve the WARP-appended `_<hash>` suffix on `SPECIES_BASE` via
`find m/species -name "${SPECIES_BASE}_*"`, but export scripts deliberately fail
when that pattern matches multiple directories. In that case, set
`SPECIES_BASE` to the full `<species>_<hash>` directory name.

### Variable naming (consistent with the rest of the skill)

| Variable here | Meaning | Source |
|---|---|---|
| `$ANGPIX` | unbinned pixel size | gather / `create_settings` |
| `$OUTPUT_ANGPIX` | target pixel size for re-exported subtomos | user choice |
| `$COORDS_ANGPIX` | pixel size of coords in the refined STAR (= unbinned `$ANGPIX` for MCore output) | refinement output |
| `$SUBTOMO_BOX_SIZE`, `$DIAMETER` | particle box / diameter | gather |
| `$SPECIES_BASE` | species name (without WARP's `_<hash>` suffix) | user choice in step 3 |
| `$WORK_DIR` | project root | gather |

---

## OPUS-ET Subtomogram Averaging (Fixed Latent Code)

For homogeneous averaging — fixes z=0 for all particles, no heterogeneity modeling.

```bash
torchrun --nproc_per_node=4 -m cryodrgn.commands.train_tomo_dist \
    <particles.star> \
    --poses <pose.pkl> \
    -n 60 \
    -b 4 \
    --zdim 16 \
    --lr 3.e-5 \
    --num-gpus 4 \
    --multigpu \
    --beta-control 0.5 \
    -o <output_dir> \
    -r <mask.mrc> \
    --split <output_dir>/deep_splt.pkl \
    --lamb 0.5 \
    --bfactor 4. \
    --valfrac 0.0 \
    --templateres 128 \
    --tmp-prefix tmp \
    --datadir <warp_tiltseries> \
    --angpix $ANGPIX \
    --downfrac 1. \
    --warp \
    --ctfalpha 0. \
    --ctfbeta 1 \
    --encode-mode fixed \
    --tilt-range $TILT_RANGE \
    --tilt-step $TILT_STEP \
    --checkpoint 10
```

**`--encode-mode fixed`:** disables encoder, all particles share z=0 → single homogeneous average.

| Parameter | Averaging | Heterogeneity |
|-----------|-----------|---------------|
| `--encode-mode` | `fixed` | `conv` or `resnet` |
| `--valfrac` | `0.0` | `0.1` |
| `--poses` | required | optional (use `--estpose`) |
| Output | single average | heterogeneous ensemble |

**When to use:** final homogeneous average after clustering/classification, or traditional STA workflow.
