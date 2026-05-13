#!/usr/bin/env python3
"""
Extract candidates from Template Matching results in parallel.

This script uses multiprocessing to run extractCandidates.py on multiple
tomograms simultaneously for faster processing.
"""

import os
import sys
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
from glob import glob


def _suffix_from_score(score_file):
    """Return the template suffix from scores_<suffix>.em."""
    base = os.path.basename(score_file)
    if not base.startswith("scores_") or not base.endswith(".em"):
        return None
    return base[len("scores_"):-len(".em")]


def find_score_angle_pairs(result_dir):
    """Return matched scores_*.em / angles_*.em pairs in one result directory."""
    pairs = []
    for scores_file in sorted(glob(os.path.join(result_dir, "scores_*.em"))):
        suffix = _suffix_from_score(scores_file)
        if not suffix:
            continue
        angles_file = os.path.join(result_dir, f"angles_{suffix}.em")
        if os.path.exists(angles_file):
            pairs.append((suffix, scores_file, angles_file))
    return pairs


def find_scores_angles(job_dir, template_name=None):
    """Find scores and angles files in a job directory.

    PyTom layouts vary by how the TM jobs were organized:
    - one job directory can contain scores_<template>.em / angles_<template>.em
    - a job directory can contain per-template subdirectories with those files

    If template_name is provided (e.g. "reference19" or "ribo"), first look
    for scores_<template_name>.em / angles_<template_name>.em in job_dir, then
    look in job_dir/<template_name>/. Falls back to the first available file
    only when template_name is None, warning if ambiguous.
    """
    if template_name:
        search_dirs = [job_dir]
        template_subdir = os.path.join(job_dir, template_name)
        if os.path.isdir(template_subdir):
            search_dirs.append(template_subdir)

        for result_dir in search_dirs:
            scores_file = os.path.join(result_dir, f"scores_{template_name}.em")
            angles_file = os.path.join(result_dir, f"angles_{template_name}.em")
            if os.path.exists(scores_file) and os.path.exists(angles_file):
                return scores_file, angles_file

        if os.path.isdir(template_subdir):
            pairs = find_score_angle_pairs(template_subdir)
            if len(pairs) == 1:
                _, scores_file, angles_file = pairs[0]
                return scores_file, angles_file

        raise FileNotFoundError(
            f"Template '{template_name}' not found in {job_dir}. "
            f"Available: {glob(os.path.join(job_dir, 'scores_*.em')) + glob(os.path.join(job_dir, '*', 'scores_*.em'))}"
        )

    # No template specified — use first available (warn if multiple exist).
    pairs = find_score_angle_pairs(job_dir)
    if not pairs:
        for subdir in sorted(glob(os.path.join(job_dir, "*"))):
            if os.path.isdir(subdir):
                pairs.extend(find_score_angle_pairs(subdir))
    if not pairs:
        raise FileNotFoundError(f"No matched scores_*.em / angles_*.em pair found in {job_dir}")
    if len(pairs) > 1:
        print(f"  WARNING: multiple templates in {job_dir}, using {os.path.basename(pairs[0][1])}. "
              f"Use --template to select explicitly.")
    return pairs[0][1], pairs[0][2]


def find_tm_results(tm_base_dir, template_name=None):
    """Find all template matching result directories."""
    job_dirs = []
    pattern = os.path.join(tm_base_dir, "*/job.xml")

    for job_xml in sorted(glob(pattern)):
        job_dir = os.path.dirname(job_xml)
        job_name = os.path.basename(job_dir)

        try:
            scores_file, angles_file = find_scores_angles(job_dir, template_name)
        except FileNotFoundError as e:
            print(f"  Skipping {job_name}: {e}")
            continue

        job_dirs.append({
            'name': job_name,
            'dir': job_dir,
            'job_xml': job_xml,
            'scores': scores_file,
            'angles': angles_file,
        })

    return job_dirs


def extract_candidates(job_info, particle_count, mask_radius, min_score, output_path):
    """Extract candidates from a single tomogram."""
    job_name = job_info['name']
    job_xml = job_info['job_xml']
    scores_file = job_info['scores']
    angles_file = job_info['angles']
    
    # Create output filename
    particle_xml = os.path.join(output_path, f"{job_name}_particles.xml")
    
    # Create log file for this job
    log_file = os.path.join(output_path, f"{job_name}_extraction.log")
    
    print(f"[{job_name}] Starting extraction...")
    print(f"  Job: {job_xml}")
    print(f"  Scores: {scores_file}")
    print(f"  Angles: {angles_file}")
    print(f"  Log: {log_file}")
    
    # Build command
    cmd = [
        'extractCandidates.py',
        '--jobFile', job_xml,
        '--result', scores_file,
        '--orientation', angles_file,
        '--particleList', particle_xml,
        '--particlePath', output_path,
        '--size', str(mask_radius),
        '--numberCandidates', str(particle_count),
        '--minimalScoreValue', str(min_score)
    ]
    
    try:
        # Open log file and run command with real-time output
        with open(log_file, 'w') as log_f:
            log_f.write(f"Extracting candidates for {job_name}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write("="*60 + "\n\n")
            log_f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Stream output to log file and also print progress
            assert process.stdout is not None
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()
                # Print progress lines to console
                if 'progress' in line.lower() or 'candidate' in line.lower() or '%' in line:
                    print(f"[{job_name}] {line.rstrip()}")
            
            # Wait for completion
            process.wait(timeout=14400)  # 4 hour timeout
            
            if process.returncode == 0:
                print(f"[{job_name}] ✓ Success: {particle_xml}")
                log_f.write("\nExtraction completed successfully\n")
                return {'name': job_name, 'status': 'success', 'output': particle_xml}
            else:
                print(f"[{job_name}] ✗ Failed: exit code {process.returncode}")
                log_f.write(f"\nExtraction failed with exit code {process.returncode}\n")
                return {'name': job_name, 'status': 'failed', 'error': f'exit code {process.returncode}'}
            
    except subprocess.TimeoutExpired:
        print(f"[{job_name}] ✗ Timeout after 4 hours")
        with open(log_file, 'a') as log_f:
            log_f.write("\nTimeout after 4 hours\n")
        return {'name': job_name, 'status': 'timeout'}
    except Exception as e:
        print(f"[{job_name}] ✗ Exception: {e}")
        with open(log_file, 'a') as log_f:
            log_f.write(f"\nException: {e}\n")
        return {'name': job_name, 'status': 'error', 'error': str(e)}


def extract_worker(args):
    """Worker function for multiprocessing pool."""
    job_info, particle_count, mask_radius, min_score, output_path = args
    return extract_candidates(job_info, particle_count, mask_radius, min_score, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Extract candidates from template matching results in parallel'
    )
    parser.add_argument(
        '--tm-dir', 
        default='template_matching/jobs',
        help='Base directory with template matching job outputs'
    )
    parser.add_argument(
        '--output', 
        default='particles',
        help='Output directory for particle lists'
    )
    parser.add_argument(
        '-n', '--num-candidates',
        type=int,
        default=5000,
        help='Number of candidates to extract per tomogram (default: 5000)'
    )
    parser.add_argument(
        '--mask-radius',
        type=int,
        default=12,
        help='Particle mask radius in pixels (default: 12)'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.0,
        help='Minimum correlation score threshold (default: 0.0)'
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=None,
        help='Number of parallel jobs (default: number of CPUs)'
    )
    parser.add_argument(
        '--template',
        default=None,
        help='Reference/template name used by PyTom outputs (e.g. "reference19", '
             '"ribo", "26s"). The extractor resolves either scores_<name>.em / '
             'angles_<name>.em in the job directory or a job_dir/<name>/ result '
             'subdirectory. Required when multiple templates are present. Omit '
             'only for confirmed single-reference jobs.'
    )

    args = parser.parse_args()

    # Get absolute paths
    tm_base_dir = os.path.abspath(args.tm_dir)
    output_path = os.path.abspath(args.output)
    
    print("=" * 60)
    print("Extract Candidates from Template Matching (Parallel)")
    print("=" * 60)
    print()
    print(f"TM directory: {tm_base_dir}")
    print(f"Output path: {output_path}")
    print(f"Candidates per tomogram: {args.num_candidates}")
    print(f"Mask radius: {args.mask_radius}")
    print(f"Min score: {args.min_score}")
    print()
    
    # Find all TM results
    print("Finding template matching results...")
    if args.template:
        print(f"Template filter: {args.template}")
    job_dirs = find_tm_results(tm_base_dir, template_name=args.template)
    
    if not job_dirs:
        print(f"ERROR: No template matching results found in {tm_base_dir}")
        sys.exit(1)
    
    print(f"Found {len(job_dirs)} tomogram(s)")
    print()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Determine number of parallel jobs
    num_jobs = args.jobs if args.jobs else min(cpu_count(), len(job_dirs))
    print(f"Running with {num_jobs} parallel process(es)")
    print()
    
    # Prepare arguments for pool
    pool_args = [
        (job, args.num_candidates, args.mask_radius, args.min_score, output_path)
        for job in job_dirs
    ]
    
    # Run extraction in parallel
    print("=" * 60)
    print("Starting extraction...")
    print("=" * 60)
    print()
    
    results = []
    with Pool(processes=num_jobs) as pool:
        results = pool.map(extract_worker, pool_args)
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print()
    print("=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"Total: {len(results)} tomogram(s)")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {failed_count}")
    print()
    print(f"Output files: {output_path}/*_particles.xml")
    print()
    print("Next steps:")
    print("  1. Inspect particle lists in PyTom GUI")
    print("  2. Filter by correlation score")
    print("  3. Convert to star file for WARP/OPUS-ET")
    print("=" * 60)
    
    # Return exit code based on results
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()
