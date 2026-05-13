# Gotchas and Script Design Patterns

---

## WARP Gotchas

### CTF range max exceeds Nyquist
- **Symptom**: `Error: Max frequency to fit is higher than the Nyquist frequency`
- **Cause**: `--range_max` / `--c_range_max` too high for the pixel size
- **Fix**: Must be ≥ 2× angpix. For 2.17 Å, Nyquist ≈ 4.35 Å → use at least 5 Å

### Missing LD_LIBRARY_PATH causes GLIBCXX errors
- Set `export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"` before running WarpTools
- Missing this causes `version 'GLIBCXX_3.4.20' not found`
- Always set `ulimit -v unlimited` to avoid virtual memory cap

### Tomogram dimensions must be UNBINNED
- `TS_XXX_ali.mrc` from AreTomo is **binned** (e.g., 8× or 16×)
- WARP `--tomo_dimensions` requires **unbinned** pixel count
- Use `headerPyTom TS_XXX_ali.mrc` to check binned dims, then multiply by binning factor
- **awk gotcha**: `$(NF-2)` not `$NF-2` when parsing from the end:
  ```bash
  nx=$(echo "$header_output" | awk '{print $(NF-2)}')  # CORRECT
  nx=$(echo "$header_output" | awk '{print $NF-2}')    # WRONG — subtracts 2 from value
  ```

### Wrong tilt angle convention (WARP ↔ AreTomo)
- WARP: +max descending to -min (e.g., +54, +51, …, 0, …, -54)
- AreTomo: -max ascending to +min (e.g., -54, -51, …, 0, …, +54)
- Fix: `awk '{print -$1}'` negates each angle; the order is preserved in the respective sign convention. Negate output back for WARP import.

### OutOfMemoryException
- Use 256 GB+ RAM in SLURM jobs
- Reduce `--perdevice` to 1
- All scripts include `ulimit -v unlimited`

---

## AreTomo Gotchas

### Job timeout
- AreTomo on many tilt series can run 12–24+ hours
- Increase `#SBATCH --time=48:00:00` for large datasets
- Consider splitting into batches

---

## Template Matching Gotchas

### Disable `set -e` in processing loops
- `set -e` causes bash to exit on any failure — including mid-loop
- Comment it out when using manual error handling:
  ```bash
  # set -e  # disabled — we handle errors manually
  for file in ...; do
      result=$(command "$file") || true
      if [ -z "$result" ]; then continue; fi
  done
  ```

### Run TM jobs sequentially to avoid GPU OOM
- Template matching is memory-intensive; max 2 concurrent jobs on 8-GPU node
- `run_tm_sequential.slurm` handles sequential execution automatically

### Match SLURM tasks to MPI processes
```bash
#SBATCH --ntasks=4           # must match mpiexec -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
```

### Estimated time: ~5–10 minutes per tomogram
- Depends on template size, tomogram size, angle count (e.g., 7112 for `angles_12.85_7112.em`)
- Example: 30 tomograms ≈ 3–5 hours → use `--time=6:00:00`

### Multiple templates in same job directory
- PyTOM writes `scores_reference19.em` and `angles_reference19.em` per template
- Using `glob("scores_*.em")[0]` picks an arbitrary one
- Some job organizations put per-template results in a subdirectory instead;
  use the same reference/template name either way.
- **Fix**: Always pass explicit `--template reference19` to extraction scripts:
  ```python
  # GOOD
  scores_file = job_dir / f"scores_{template_name}.em"
  # BAD
  scores_files = list(job_dir.glob("scores_*.em"))  # arbitrary!
  ```

### PyTOM subregion attribute requires spaces
```xml
<!-- CORRECT — spaces inside quotes -->
<Volume Filename="tomo.mrc" Subregion=" 0,0,20,1000,1000,460 "/>
<!-- WRONG — no spaces → PyTOM parse error -->
<Volume Filename="tomo.mrc" Subregion="0,0,20,1000,1000,460"/>
```
Format: `" x_start,y_start,z_start,x_dim,y_dim,z_dim "` (leading + trailing space)

### Use AreTomo tomograms for TM, not WARP reconstructions
- `*_ali.mrc` from AreTomo has better contrast for template matching
- WARP's `reconstruction/*.mrc` is for visual inspection and downstream analysis

### dsdsh convert_pytom output naming
- Output name is based on **tilt series name argument**, not the input filename:
  ```bash
  dsdsh convert_pytom TS_026.star TS_026  # → TS_026_norm.star (NOT TS_026.star_norm.star)
  ```
- Changes to current working directory for output
- **Safe pattern**:
  ```bash
  cd "$OUTPUT_DIR"
  dsdsh convert_pytom "$star_file" "$ts_name"
  mv "${ts_name}_norm.star" "${ts_name}_warp.star"
  ```

### Extracting tilt series names — exact prefix match
- Names like `Position_2_Position_2_ali.star` need careful extraction
- **Wrong**: `sed 's/_.*$//'` → gives `Position` (loses the number)
- **Correct**: `grep -oP '^[A-Za-z]+_[0-9]+'` → gives `Position_2`

### File overwriting in multi-step conversions
- `convert.py` and `dsdsh` may not overwrite existing files cleanly
- Always remove output files before re-running:
  ```bash
  rm -f "$star_file" "$warp_star" "${OUTPUT_DIR}/${ts_name}_norm.star"
  convert.py -f input.xml ...
  ```

---

## OPUS-ET Training Gotchas

### --split flag is required and must use absolute path
```bash
# CORRECT
OUTPUT_DIR="$WORK_DIR/z12ribo"
SPLIT_FILE="$OUTPUT_DIR/deep.pkl"   # in output dir, absolute path
torchrun ... --split "$SPLIT_FILE" -o "$OUTPUT_DIR"

# WRONG — relative paths break across cwd changes
torchrun ... --split deep.pkl -o z12ribo
```

### Check tilt angles before setting TILT_RANGE and TILT_STEP
```bash
head -1 warp_tiltseries/tiltstack/TS_XXX/TS_XXX.tlt  # max angle (e.g., 54)
head -3 warp_tiltseries/tiltstack/TS_XXX/TS_XXX.tlt  # first 3 to compute step
```
Default in scripts: `TILT_RANGE=50`, `TILT_STEP=2` — always verify from actual `.tlt` files.

---

## Script Design Patterns

### Canonical tilt series loop — always use tomostar files as source

Tomostar files represent tilt series successfully imported into WARP. Loop over them rather than over MRC files directly.

```bash
TOMOSTAR_DIR="$WORK_DIR/tomostar"
TOMO_DIR="$WORK_DIR/warp_tiltseries/tiltstack"

mapfile -t tomostar_files < <(find "$TOMOSTAR_DIR" -name "*.tomostar" -type f | sort)

PROCESSED=0; FAILED=0

for tomostar_file in "${tomostar_files[@]}"; do
    ts_name=$(basename "$tomostar_file" .tomostar)

    # Find aligned tomogram with wildcard (use find, not glob variable)
    tomo_file=$(find "$TOMO_DIR/$ts_name" -name "*_ali.mrc" -type f -print -quit 2>/dev/null)

    if [ -z "$tomo_file" ]; then
        echo "Skipping $ts_name — aligned tomogram not found"
        ((FAILED++)); continue
    fi

    # Process...
    ((PROCESSED++))
done

echo "Done: $PROCESSED processed, $FAILED failed"
```

**Anti-patterns to avoid:**
```bash
# DON'T — loops over MRC directly (may include non-tomostar files)
for tomo in "$TOMO_DIR"/*/*.mrc; do ...

# DON'T — while read creates a subshell, counter updates are lost
find "$TOMOSTAR_DIR" -name "*.tomostar" | while read f; do
    ((PROCESSED++))  # this won't persist!

# DON'T — glob in variable isn't expanded by bash
PATTERN="*.mrc"
for f in "$DIR/$PATTERN"; do  # looks for a literal file named "*.mrc"
```

### Exact name matching — avoid substring collisions

`Position_1` must not match `Position_12`. Always use underscore suffix:

```bash
# BAD
if [[ "$subdir" == *"$ts_name"* ]]; then   # Position_1 matches Position_12!

# GOOD
if [[ "$subdir" == "${ts_name}_"* ]]; then  # Position_1_ only matches Position_1_*

# GOOD (Python)
if subdir.name.startswith(f"{ts_name}_"):
```

### Parallel processing with Python multiprocessing

For CPU-bound tasks on multiple tilt series, use Python's `multiprocessing.Pool` rather than bash loops:
- Better error handling and per-tilt-series log files
- Progress tracking with ✓/✗ status
- `--dry-run` capability
- See `scripts/extract_candidates_parallel.py` as a reference implementation

```python
def process_one(args_tuple):
    ts_name, ... = args_tuple
    # do work, return (ts_name, success, message)

with mp.Pool(processes=num_workers) as pool:
    results = pool.map(process_one, jobs)

for ts_name, success, msg in sorted(results):
    print(f"[{'✓' if success else '✗'}] {ts_name}: {msg}")
```

SLURM wrapper calls the Python script with `--jobs N` to set pool size.

| Task | Parallel method |
|------|-----------------|
| Template matching (PyTOM) | Sequential — GPU memory limited |
| Candidate extraction | Python multiprocessing |
| STAR conversion | Python multiprocessing |

---

## Resuming after a failed or killed job

The one place that needs explicit thought is `train_opuset` /
`train_opuset_fixed`: without `--load <weights.N.pkl>`, the script restarts
from epoch 0 and discards prior progress. Edit the script to add `--load`
before re-submitting.

For other phases, check what the relevant tool does on re-run before bulk
re-submitting (`WarpTools <command> --help`, AreTomo log behavior, etc.) —
behavior depends on the tool and the WARP version, and assumptions baked into
this skill may not hold on every install.

### Defocus handedness — pick the right flag

`WarpTools ts_defocus_hand` has five mutually exclusive modes (see WARP API
docs: <https://warpem.github.io/warp/reference/warptools/api/tilt_series/>):

| Flag | Behavior | Idempotent? |
|---|---|---|
| `--check` | Print correlation only; change nothing | yes (read-only) |
| `--set_flip` | Set handedness to "flip" for all tilt series | yes (absolute) |
| `--set_noflip` | Set handedness to "no flip" for all tilt series | yes (absolute) |
| `--set_auto` | Check correlation, then set the appropriate value automatically | yes (absolute) |
| `--set_switch` | Invert each tilt series' current value | **no — toggle** |

Recommended flow:

- **Easiest**: just use `--set_auto` once. Single submission, no manual
  decision based on correlation sign.
- **Manual**: run `--check` first, then re-run with `--set_flip` (negative
  correlation) or `--set_noflip` (positive correlation).
- **Avoid `--set_switch`** unless you genuinely want to toggle state — running
  it twice undoes itself, which is the only re-`sbatch` hazard in this phase.
