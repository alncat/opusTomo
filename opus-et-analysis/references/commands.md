# opus-et Commands Reference

## Analysis Commands

### dsdsh analyze
Run PCA and kmeans clustering on a specific epoch.

```bash
dsdsh analyze <workdir> <epoch> <numpc> <numk>
```

**Parameters:**
- `workdir` - Working directory with weights.N.pkl, z.N.pkl files
- `epoch` - Epoch number to analyze (e.g., 39)
- `numpc` - Number of PCA components
- `numk` - Number of kmeans clusters

**Output:**
- Creates `analyze.<epoch>/` directory with:
  - `kmeans<numk>/` - Cluster labels, centers, star files
  - `pc<N>/` - Principal component trajectory files
  - UMAP plots and PCA plots

### dsd eval_vol
Generate volumes from latent codes.

```bash
dsd eval_vol --load weights.<epoch>.pkl \
    -c config.pkl \
    -o <outdir> \
    --zfile <z_values.txt> \
    --Apix <apix> \
    --prefix <prefix>
```

**Common uses:**
- Kmeans centers: `--zfile analyze.<epoch>/kmeans<numk>/centers.txt`
- PC traversal: `--zfile analyze.<epoch>/pc<N>/z_pc.txt`

### dsd parse_pose_star
Parse poses from star file. Two modes:

**Mode 1: Split by cluster labels (generates multiple star files)**
```bash
dsd parse_pose_star <starfile> \
    -D <box_size> \
    --Apix <apix> \
    --labels <labels.pkl> \
    --outdir <outdir>
```

**Mode 2: Generate pose pickle (single output)**
```bash
dsd parse_pose_star <starfile> \
    -D <box_size> \
    --Apix <apix> \
    -o <output_pose.pkl>
```

**Important:** Use `effective_box_size = lattice_args['D'] - 1` for `-D` parameter.

### dsdsh combine_star
Combine two star files into one.

```bash
dsdsh combine_star <starfile1> <starfile2> <output.star>
```

For multiple files, chain the commands:
```bash
dsdsh combine_star a.star b.star temp.star
dsdsh combine_star temp.star c.star output.star
```

## File Locations from Config

From `config.pkl`:
- `config['model_args']['Apix']` - Training-effective pixel size (NOT the original)  
- `config['dataset_args']['downfrac']` - Downsampling fraction
- **Original pixel size** = `Apix * downfrac` — use this for `parse_pose_star` and all original-data operations
- `config['lattice_args']['D']` - Lattice size (use D-1 for actual box size)
- `config['dataset_args']['particles']` - Original star file path
- `config['dataset_args']['poses']` - Pose pickle file path
- `config['model_args']['zdim']` - Latent dimension

The original pixel size can also be read from the star file's `_rlnDetectorPixelSize` column.

## Common Analysis Workflow

1. **Extract config:** `python <skill-dir>/scripts/extract_config.py config.pkl`  
   Note the `original_Apix` and `effective_box_size` values.
2. **Analyze epoch:** `dsdsh analyze . 39 10 20`
3. **Generate kmeans volumes:** `dsd eval_vol --load weights.39.pkl -c config.pkl -o kmeans_volumes --zfile analyze.39/kmeans20/centers.txt --Apix <original_apix> --prefix kmeans_cluster`
4. **Generate PC volumes:** `dsd eval_vol ... --zfile analyze.39/pc5/z_pc.txt --prefix pc5`
5. **Create cluster star files:** `dsd parse_pose_star ribotm.star -D <effective_box_size> --Apix <original_apix> --labels analyze.39/kmeans20/labels.pkl --outdir kmeans_pose`
6. **Combine clusters:** `dsdsh combine_star pre9.star pre10.star combined.star`
7. **Generate pose pickle for combined:** `dsd parse_pose_star combined.star -D <effective_box_size> --Apix <original_apix> -o combined_pose.pkl`
