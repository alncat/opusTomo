# WARP/OPUS-ET Complete Workflow

## Test on a small subset first (long-running phases only)

For phases that take many hours per submission, validate the configuration on
a 1–3 tilt-series subset before launching the full dataset. The phases worth
subset-testing are:

- **Phase 3 — `warp_export_stacks` + AreTomo alignment** (~5–15 min/TS)
- **Phase 5 — `warp_ts_reconstruct`** (~5–10 min/TS at moderate binning)
- **Phase 6 — template matching** (~5–10 min/TS)

Cheap phases (frame-series import, tilt-series setup, settings updates,
CTF estimation, alignment import, mask generation, STAR conversion) finish in
seconds to minutes — run them on the full set and inspect the output.

To subset-test, set `MAX_TS=N` (e.g. `MAX_TS=2`) in the CONFIGURATION block of
the script you want to validate. The three loops support this cap:

- `warp_aretomo_align_negate.slurm` — limits the alignment loop
- `gen_tm_jobs_aretomo.slurm` — limits the job-XML generation loop
- `run_tm_sequential.slurm` — limits how many TM jobs actually run

After the subset run looks correct, set `MAX_TS=0` to disable the cap and
re-submit for the full set. A common-enough mistake (wrong `BINNING_FACTOR`,
wrong `TOMO_DIMS`, wrong `ALIGN_ANGPIX`, wrong template path) caught on 2 TS
in 10 minutes saves a 24-hour job from finishing with bad parameters.

Note: `warp_ts_reconstruct.slurm` and `warp_ts_ctf.slurm` use a single
`WarpTools` call that processes every tilt series in the settings file, so
they don't have a `MAX_TS` knob. To subset-test those, point them at a
trimmed `tomostar/` (e.g. `tomostar_test/`) and a separate `.settings` file.

## Phase 1: Frame Series Processing

### 1.1 Create Frame Series Settings
```bash
WarpTools create_settings \
    --folder_data ../metadata/movies/ \
    --folder_processing warp_frameseries \
    --extension "*.mrc" \
    --angpix 3.37 \
    --exposure 2.36 \
    --output warp_frameseries.settings
```

### 1.2 Link Tilt Images (Single-Frame)
For single-frame tilt images (skip motion correction):
```bash
mkdir -p warp_frameseries/average
for f in ../metadata/movies/*.mrc; do
    ln -sf $(realpath $f) warp_frameseries/average/$(basename $f)
done
```

### 1.3 Estimate CTF on Frame Series
```bash
WarpTools fs_ctf \
    --settings warp_frameseries.settings \
    --grid 2x2x1 \
    --range_max 7 \
    --defocus_max 8
```

### 1.4 (Alternative) Use SLURM Script for Frame Series
```bash
sbatch scripts/warp_frameseries_import.slurm
```
This script automates steps 1.1-1.3 with configurable parameters.

## Phase 2: Tilt Series Setup

### 2.1 Create Tilt Series Settings
```bash
WarpTools create_settings \
    --folder_data tomostar \
    --folder_processing warp_tiltseries \
    --extension "*.tomostar" \
    --angpix 3.37 \
    --exposure 2.36 \
    --tomo_dimensions 3840x3712x2000 \
    --output warp_tiltseries.settings
```

### 2.2 Import Tilt Series from MDOC
```bash
WarpTools ts_import \
    --mdocs mdoc/ \
    --frameseries warp_frameseries \
    --tilt_exposure 2.36 \
    --min_intensity 0.3 \
    --output tomostar
```

### 2.3 (Alternative) Use SLURM Script for Tilt Series Setup
```bash
sbatch scripts/warp_tiltseries_setup.slurm
```
This script automates steps 2.1-2.2 with configurable parameters.

## Phase 3: Tilt Series Alignment

### 3.1 Export Tilt Stacks for AreTomo
```bash
# 4x binning: 3.37 * 4 = 13.48 Å
WarpTools ts_stack \
    --settings warp_tiltseries.settings \
    --angpix 13.48
```

Output: `warp_tiltseries/tiltstack/TS_XXX/TS_XXX.st` and `TS_XXX.rawtlt`

### 3.2 (Alternative) Use SLURM Script for Export
```bash
sbatch scripts/warp_export_stacks.slurm
```
This script exports tilt stacks with automatic pixel size calculation from settings.

### 3.3 Run AreTomo2 Alignment
```bash
AreTomo2 \
    -InMrc TS_026.st \
    -OutMrc TS_026_ali.mrc \
    -OutBin 1 \
    -AngFile TS_026.rawtlt \
    -OutImod 2 \
    -VolZ 256 \
    -Gpu 0 \
    -AlignZ 100 \
    -FlipVol 1 \
    -FlipInt 1 \
    -DarkTol 0.1 \
    -Wbp 1
```

### 3.4 (Alternative) Use SLURM Script for AreTomo2 + WARP Prep

**Option A: Angle negation method (recommended for consistency)**
```bash
sbatch scripts/warp_aretomo_align_negate.slurm
```
This script:
1. Negates `.rawtlt` → `_neg.rawtlt` (WARP → AreTomo convention)
2. Runs AreTomo2 with negated angles
3. Negates output back (AreTomo → WARP convention)

This ensures consistent angle conversion throughout the workflow.

## Phase 4: Import Alignment Parameters

### 4.1 Import Alignments into WARP

**Manual import:**
```bash
WarpTools ts_import_alignments \
    --settings warp_tiltseries.settings \
    --alignments warp_tiltseries/tiltstack/ \
    --alignment_angpix 13.48
```

**SLURM script:**
```bash
sbatch scripts/warp_import_alignments.slurm
```

**IMPORTANT:** Use the SAME binned pixel size as used in AreTomo2 alignment (e.g., 13.48 for 4x binning). The alignment parameters (.xf, .tlt) are in binned pixel units.

## Phase 5: CTF and Reconstruction

### 5.1 Check Defocus Handedness
```bash
WarpTools ts_defocus_hand \
    --settings warp_tiltseries.settings \
    --check
```

**If average correlation is negative:**
```bash
WarpTools ts_defocus_hand \
    --settings warp_tiltseries.settings \
    --set_flip
```

### 5.2 CTF Estimation on Tilt Series

**Manual:**
```bash
WarpTools ts_ctf \
    --settings warp_tiltseries.settings \
    --window 512 \
    --range_high 7 \
    --defocus_max 8
```

**SLURM script** (includes defocus handedness check):
```bash
sbatch scripts/warp_ts_ctf.slurm
```

### 5.3 Reconstruct Tomograms

**Manual:**
```bash
# Enable 32-bit float output
export WARP_FORCE_MRC_FLOAT32=1

WarpTools ts_reconstruct \
    --settings warp_tiltseries.settings \
    --angpix 13.48
```

**SLURM script:**
```bash
sbatch scripts/warp_ts_reconstruct.slurm
```

## Phase 6: Template Matching 

### 6.1 Generate Sphere Mask
```bash
sbatch scripts/gen_sphere_mask.slurm
```

Creates a spherical mask `mask_ribo.mrc` matching the template dimensions.

### 6.2 Generate Template Matching Job XMLs
```bash
sbatch scripts/gen_tm_jobs_aretomo.slurm
```

Generates `template_matching/<TM_LABEL>/jobs/TS_XXX/job.xml` files configured to use:
- AreTomo-reconstructed tomograms (`*_ali.mrc`)
- Template and mask volumes
- Wedge angles calculated from tilt angles
- PyTom angle list (`angles_12.85_7112.em`)

### 6.3 Run Template Matching Jobs
```bash
sbatch scripts/run_tm_sequential.slurm
```

Runs all template matching jobs **sequentially** on a single node:
- Uses 4 GPUs (`--gres=gpu:4`)
- 4 MPI processes (`mpiexec -n 4`)
- One job at a time (avoids GPU memory conflicts)
- Estimated time: ~5-10 minutes per tomogram

### 6.4 Extract Particle Candidates
```bash
sbatch scripts/extract_tm_candidates_parallel.slurm
```

Extracts particle candidates from template matching results:
- Extracts **5000 particles** per tomogram by default
- Uses parallel processing (multiple CPUs)
- Output: `template_matching/<TM_LABEL>/particles/TS_XXX_particles.xml`
- Individual log files for each tomogram

Parameters (edit script):
- `NUM_CANDIDATES=5000` - particles to extract
- `MASK_RADIUS=12` - particle radius in pixels
- `MIN_SCORE=0.0` - minimum correlation score

### 6.5 Convert to STAR Format
```bash
sbatch scripts/convert_to_star.slurm
```

Converts PyTom particle XML files to STAR format:
- Output: `template_matching/<TM_LABEL>/star_files/*.star`

### 6.6 Convert to WARP Format
```bash
sbatch scripts/convert_pytom_to_warp.slurm
```

Converts PyTom STAR files to WARP-compatible format:
- Uses `dsdsh convert_pytom`
- Output: `template_matching/<TM_LABEL>/warp_star/*_warp.star`
- Output files are named `<ts_name>_norm.star` then renamed

## Phase 7: Export for OPUS-ET

### 7.1 Export Subtomograms

**Manual command:**
```bash
WarpTools ts_export_particles \
    --settings warp_tiltseries.settings \
    --input_directory template_matching/<TM_LABEL>/warp_star/ \
    --input_pattern "*_warp.star" \
    --output_star warp_tiltseries/matchingribo.star \
    --output_angpix 3.37 \
    --box 128 \
    --diameter 320 \
    --relative_output_paths \
    --3d \
    --coords_angpix 3.37 \
    --output_ctf_csv \
    --dont_correct_ctf_3d
```

**SLURM script:**
```bash
sbatch scripts/warp_export_particles.slurm
```

Required flags for OPUS-ET compatibility:
- `--output_ctf_csv` - Outputs CTF parameters
- `--dont_correct_ctf_3d` - OPUS-ET handles CTF correction

### 7.2 Prepare OPUS-ET Input
```bash
dsdsh prepare warp_tiltseries/matchingribo.star 3.37 128
```

Or manually create pose pickle:
```bash
dsd parse_pose_star warp_tiltseries/matchingribo.star \
    -D 128 --Apix 3.37 -o particles_pose.pkl
```

## Phase 8: OPUS-ET Training

### 8.1 Generate Training Mask (Optional)
```bash
sbatch scripts/gen_training_mask.slurm
```

Creates a larger mask (`mask_ribo_training.mrc`) sized for the 128³ training box.

### 8.2 Train OPUS-ET Model
```bash
sbatch scripts/train_opuset.slurm
```

Trains the heterogeneity analysis model:
- 4 GPUs, 40 epochs, ~5 days runtime
- Output: `z12ribo/` directory with model weights
- Key parameters: `ZDIM=12`, `BATCH_SIZE=12`, `LEARNING_RATE=4.5e-5`

### 8.3 Analyze Results
```bash
# Analyze epoch 39 (final epoch)
dsdsh analyze z12ribo 39 10 20

# Generate volumes for k-means clusters
dsd eval_vol --load z12ribo/weights.39.pkl \
    -c z12ribo/config.pkl \
    -o kmeans_volumes \
    --zfile z12ribo/analyze.39/kmeans20/centers.txt \
    --Apix 3.37 --prefix kmeans_cluster
```

## Quick SLURM Submission

For automated processing, use:
```bash
# Phase 1: Frame series import + CTF
sbatch scripts/warp_frameseries_import.slurm

# Phase 2: Tilt series setup (settings + MDOC import)
sbatch scripts/warp_tiltseries_setup.slurm

# Phase 3: Export tilt stacks for alignment
sbatch scripts/warp_export_stacks.slurm

# Phase 3: Run AreTomo2 alignment + prepare for WARP
sbatch scripts/warp_aretomo_align_negate.slurm

# Phase 4: Import alignments into WARP
sbatch scripts/warp_import_alignments.slurm

# Phase 5: CTF estimation (includes defocus handedness check)
sbatch scripts/warp_ts_ctf.slurm

# Phase 5: Reconstruction
sbatch scripts/warp_ts_reconstruct.slurm

# Phase 6: Template matching (optional)
sbatch scripts/gen_sphere_mask.slurm               # Generate mask
sbatch scripts/gen_tm_jobs_aretomo.slurm           # Generate job XMLs
sbatch scripts/run_tm_sequential.slurm             # Run TM jobs
sbatch scripts/extract_tm_candidates_parallel.slurm # Extract particles
sbatch scripts/convert_to_star.slurm               # Convert to STAR
sbatch scripts/convert_pytom_to_warp.slurm         # Convert to WARP format

# Phase 7: Export for OPUS-ET
sbatch scripts/warp_export_particles.slurm         # Export subtomograms

# Phase 8: OPUS-ET Training
sbatch scripts/gen_training_mask.slurm             # Generate training mask
sbatch scripts/train_opuset.slurm                  # Train model
```

## File Organization

Names like `z12ribo/`, `mask_ribo_training.mrc`, and `<species>` below are
illustrative — substitute with the user's chosen `OUTPUT_DIR` / template name
/ `SPECIES_BASE`. Epoch numbers shown as `<N-1>` resolve to `NUM_EPOCHS - 1`
(e.g. 39 for the default 40 epochs).

```
project/                              # = $WORK_DIR
├── *_<jobid>.out / *_<jobid>.err    # SLURM logs land HERE if sbatch is run
│                                       from $WORK_DIR (recommended)
├── mdoc/                             # raw MDOC files
├── tomostar/                         # canonical tilt-series list (*.tomostar)
├── tomostar_test/                    # OPTIONAL: 1–3 .tomostar copies for
│                                       subset testing (see top of this file)
│
├── warp_frameseries/                 # Phase 1: Frame series
│   ├── average/                     # linked / motion-corrected tilt images
│   ├── powerspectrum/               # CTF power spectra
│   └── TS_XXX_XX.xml                # per-image frame metadata
│
├── warp_tiltseries/                  # Phases 2–7
│   ├── tiltstack/TS_XXX/            # Phase 3: stacks + AreTomo output
│   │   ├── TS_XXX.st                # exported tilt stack
│   │   ├── TS_XXX.rawtlt            # raw angles
│   │   ├── TS_XXX_neg.rawtlt        # negated angles (AreTomo input)
│   │   ├── TS_XXX_ali.mrc           # AreTomo aligned tomogram (binned, at ALIGN_ANGPIX)
│   │   ├── TS_XXX.xf                # transforms (negated → WARP convention)
│   │   ├── TS_XXX.tlt               # angles (negated → WARP convention)
│   │   └── TS_XXX_Imod/             # AreTomo2 IMOD output
│   │       ├── TS_XXX_st.tlt        # original refined angles
│   │       └── TS_XXX_st.xf         # original transforms
│   ├── reconstruction/              # Phase 5: WARP tomograms (TS_XXX_<apix>Apx.mrc, at ALIGN_ANGPIX)
│   ├── subtomo/                     # Phase 7: ts_export_particles output
│   │   ├── TS_XXX/*.mrc             #   per-tilt-series subtomograms (at OUTPUT_ANGPIX)
│   │   └── *.csv                    #   per-particle CTF metadata (--output_ctf_csv)
│   ├── <TM_LABEL>_matching.star     # Phase 7: per-template export STAR (one per TM_LABEL)
│   ├── logs/                        # WARP processing logs
│   └── TS_XXX.xml                   # per-tilt-series metadata
│
├── templates/                        # Template generation output
│   ├── <TM_LABEL>_tm.mrc            # template at ALIGN_ANGPIX (gen_template_from_mrc.slurm)
│   └── <TM_LABEL>_mask.mrc          # sphere mask paired with the template (gen_sphere_mask.slurm)
│
├── template_matching/
│   └── <TM_LABEL>/                   # Phase 6: one namespace per species/template
│       ├── jobs/TS_XXX/             # PyTOM TM job dirs
│       │   ├── job.xml               # PyTom job config
│       │   ├── scores_<TM_LABEL>.em  # correlation scores
│       │   └── angles_<TM_LABEL>.em  # orientation angles
│       ├── particles/                # extracted candidates
│       │   ├── TS_XXX_particles.xml  # per-TS particle list (PyTom XML)
│       │   └── TS_XXX_extraction.log
│       ├── star_files/               # PyTom XML → per-TS STAR
│       │   └── TS_XXX.star
│       └── warp_star/                # WARP-compatible STAR
│           └── TS_XXX_warp.star
│
├── <output_dir>/                     # Phase 8: OPUS-ET training output (e.g. z12ribo/)
│   ├── weights.<N-1>.pkl            # final model weights (NUM_EPOCHS-1)
│   ├── z.<N-1>.pkl                  # final latent embeddings (NUM_EPOCHS-1)
│   ├── config.pkl                   # model configuration
│   ├── run.log                      # training log
│   └── analyze.<N-1>/               # analysis output
├── <output_dir>_subset1/             # Phase 8 fixed-mode on rlnRandomSubset=1 → half1
├── <output_dir>_subset2/             # Phase 8 fixed-mode on rlnRandomSubset=2 → half2
├── <template>_training_mask.mrc      # Phase 8: training-loss sphere mask
│
└── m/                                # M refinement (advanced.md)
    ├── <population_name>.population  # MTools create_population output
    └── species/
        └── <species>_<hash>/         # WARP appends an 8-char hash
            ├── <species>.species     # MTools species metadata
            ├── half1.mrc             # input half-map (from fixed-mode subset1)
            ├── half2.mrc             # input half-map (from fixed-mode subset2)
            ├── mask.mrc              # thresholded molecule mask (NOT a sphere)
            ├── <species>_filt.mrc    # filtered consensus map (MCore output)
            ├── <species>_particles.star # refined per-particle poses (MCore output, WARP STAR)
            └── <species>_relion.star    # RELION-style copy made by `dsdsh convert_warp`
                                         #   (required input to ts_export_particles)
```

## Troubleshooting

### OutOfMemoryException
- Use 256GB+ RAM
- All SLURM scripts include `ulimit -v unlimited` to remove virtual memory limits

### CTF estimation fails
- Ensure frame series CTF is done first
- Check images are linked in `warp_frameseries/average/`
- Verify 256GB+ memory available

### Job timeout / killed by SLURM
- **Symptom**: Job ends with `TIMEOUT` status, output stops mid-process, or partially completed files
- **Solution**: Increase time limit in SLURM script:
  ```bash
  #SBATCH --time=48:00:00  # Increase from default 12-24 hours
  ```
- For many tilt series or large datasets:
  - Use 48 hours or more for full pipeline
  - Process in smaller batches
  - Use separate scripts for each phase (export, align, CTF, reconstruct)
