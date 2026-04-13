#!/usr/bin/env python3
"""
Convert PyTom XML particle lists to RELION STAR format for warp-opus-tomo.

This script converts PyTom template matching results (XML particle lists) to
RELION STAR format compatible with WARP and OPUS-ET workflows.

Usage:
    python convert_pytom_to_star.py --input-dir <dir> --output <starfile> [options]
    python convert_pytom_to_star.py --input <xml_file> --output <starfile>

Examples:
    # Convert all atps35A particles from top100 tomograms
    python convert_pytom_to_star.py \\
        --input-dir atp/04_Particle_Picking/Picked_Particles \\
        --pattern "*_atps35A.xml" \\
        --output atp/atps35A_particles.star \\
        --bin-size 4 \\
        --workers 10

    # Single file conversion
    python convert_pytom_to_star.py \\
        --input particles.xml \\
        --output particles.star \\
        --bin-size 4

Output columns (RELION 3.1 format):
    _rlnImageName, _rlnMicrographName, _rlnCoordinateX/Y/Z,
    _rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi,
    _rlnOriginX/Y/Z, _rlnCtfImage, _rlnMagnification, _rlnDetectorPixelSize
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
import sys
from multiprocessing import Pool
from functools import partial
import re


def parse_xml_particle_list(xml_path):
    """
    Parse PyTom XML particle list and extract particle information.
    
    Returns list of dicts with particle data.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    particles = []
    
    for particle_elem in root.findall('Particle'):
        particle = {}
        
        # Get filename (subtomogram path)
        particle['filename'] = particle_elem.get('Filename', '')
        
        # Get rotation angles (ZXZ convention in PyTom)
        rotation = particle_elem.find('Rotation')
        if rotation is not None:
            particle['z1'] = float(rotation.get('Z1', 0))
            particle['z2'] = float(rotation.get('Z2', 0))
            particle['x'] = float(rotation.get('X', 0))
        else:
            particle['z1'] = particle['z2'] = particle['x'] = 0.0
        
        # Get shifts (origin)
        shift = particle_elem.find('Shift')
        if shift is not None:
            particle['shift_x'] = float(shift.get('X', 0))
            particle['shift_y'] = float(shift.get('Y', 0))
            particle['shift_z'] = float(shift.get('Z', 0))
        else:
            particle['shift_x'] = particle['shift_y'] = particle['shift_z'] = 0.0
        
        # Get pick position (coordinates in tomogram)
        pick_pos = particle_elem.find('PickPosition')
        if pick_pos is not None:
            particle['coord_x'] = float(pick_pos.get('X', 0))
            particle['coord_y'] = float(pick_pos.get('Y', 0))
            particle['coord_z'] = float(pick_pos.get('Z', 0))
            particle['origin'] = pick_pos.get('Origin', '')
            particle['binning'] = int(pick_pos.get('Binning', 1))
        else:
            particle['coord_x'] = particle['coord_y'] = particle['coord_z'] = 0.0
            particle['origin'] = ''
            particle['binning'] = 1
        
        # Get score
        score = particle_elem.find('Score')
        if score is not None:
            particle['score'] = float(score.get('Value', 0))
        else:
            particle['score'] = 0.0
        
        # Get wedge info (tilt angles)
        wedge = particle_elem.find('Wedge')
        if wedge is not None:
            single_tilt = wedge.find('SingleTiltWedge')
            if single_tilt is not None:
                particle['tilt_angle1'] = float(single_tilt.get('Angle1', 0))
                particle['tilt_angle2'] = float(single_tilt.get('Angle2', 0))
            else:
                particle['tilt_angle1'] = particle['tilt_angle2'] = 0.0
        else:
            particle['tilt_angle1'] = particle['tilt_angle2'] = 0.0
        
        particles.append(particle)
    
    return particles


def zxz_to_zyz(z1, z2, x):
    """
    Convert Euler angles from ZXZ convention to ZYZ convention.
    
    PyTom uses ZXZ: rotation by Z1, then X, then Z2 (intrinsic, degrees)
    RELION uses ZYZ: rotation by Rot (Z), then Tilt (Y), then Psi (Z) (intrinsic, degrees)
    
    Uses inverse transpose (r.inv()) to handle coordinate system flip,
    then converts to ZYZ Euler angles.
    
    Returns (rot, tilt, psi) in degrees.
    """
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ImportError(
            "scipy is required for Euler conversion (ZXZ -> ZYZ). "
            "Please install scipy and rerun."
        ) from exc

    # PyTom ZXZ: z1, x, z2 (intrinsic rotations about Z, X, Z axes)
    r = Rotation.from_euler('ZXZ', [z2, x, z1], degrees=True)

    # Use inverse to handle coordinate system flip (equivalent to transpose)
    r_inv = r.inv()

    # Convert to ZYZ (intrinsic)
    rot, tilt, psi = r_inv.as_euler('ZYZ', degrees=True)

    return rot, tilt, psi


def extract_tilt_series_name(origin_path):
    """
    Extract tilt series name from tomogram path.
    
    Example:
        /path/to/01082023_BrnoKrios_Arctis_WebUI_Position_40_7.84Apx.mrc
        -> 01082023_BrnoKrios_Arctis_WebUI_Position_40_7.84Apx
    """
    if not origin_path:
        return "unknown"
    
    # Get basename without extension
    base = Path(origin_path).stem
    
    # Remove common suffixes
    base = re.sub(r'_ali$', '', base)
    base = re.sub(r'_rec$', '', base)
    
    return base


def process_single_xml(args):
    """
    Process a single XML file and return particles.
    
    Args:
        args: tuple of (xml_file, index, total)
    
    Returns:
        list of particle dictionaries
    """
    xml_file, index, total = args
    xml_path = Path(xml_file)
    
    try:
        particles = parse_xml_particle_list(xml_path)
        print(f"[{index}/{total}] {xml_path.name}: {len(particles)} particles")
        return particles
    except Exception as e:
        print(f"[{index}/{total}] ERROR {xml_path.name}: {e}")
        return []


def write_star_file(particles, output_path, pixel_size=7.84, bin_size=4, magnification=10000, 
                    ctf_template=None):
    """
    Write particles to STAR file format.
    
    Parameters:
    -----------
    particles : list of dict
        Particle data from parse_xml_particle_list
    output_path : str
        Output STAR file path
    pixel_size : float
        Detector pixel size in Angstroms
    bin_size : float
        Binning factor for coordinate scaling
    magnification : float
        Magnification value
    ctf_template : str or None
        Template for CTF image path
    """
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Created by convert_pytom_to_star.py\n")
        f.write("# Particles extracted from PyTom template matching results\n")
        f.write(f"# Total particles: {len(particles)}\n")
        f.write(f"# Pixel size: {pixel_size} Å\n")
        f.write("\n")
        f.write("data_\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_rlnCoordinateX #1\n")
        f.write("_rlnCoordinateY #2\n")
        f.write("_rlnCoordinateZ #3\n")
        f.write("_rlnMicrographName #4\n")
        f.write("_rlnMagnification #5\n")
        f.write("_rlnDetectorPixelSize #6\n")
        f.write("_rlnGroupNumber #7\n")
        f.write("_rlnAngleRot #8\n")
        f.write("_rlnAngleTilt #9\n")
        f.write("_rlnAnglePsi #10\n")
        f.write("_rlnOriginX #11\n")
        f.write("_rlnOriginY #12\n")
        f.write("_rlnOriginZ #13\n")
        
        # Write particles
        for i, p in enumerate(particles, 1):
            # Convert angles from ZXZ to ZYZ
            rot, tilt, psi = zxz_to_zyz(p['z1'], p['z2'], p['x'])
            
            # Get micrograph name (origin/tomogram)
            micrograph = p['origin']
            
            # Get coordinates (simple scaling using bin_size)
            # PyTom coordinates are scaled by bin_size to get final coordinates
            coord_x = p['coord_x'] * bin_size
            coord_y = p['coord_y'] * bin_size
            coord_z = p['coord_z'] * bin_size
            
            # Get shifts (origin) - negate and scale by bin_size for RELION convention
            origin_x = -p['shift_x'] * bin_size
            origin_y = -p['shift_y'] * bin_size
            origin_z = -p['shift_z'] * bin_size
            
            # Group number (default to 1)
            group_number = 1
            
            # Write line (tab-separated) - new column order
            # CoordinateX/Y/Z, MicrographName, Magnification, DetectorPixelSize, GroupNumber, Angles, Origins
            f.write(f"{coord_x:.6f}\t{coord_y:.6f}\t{coord_z:.6f}\t")
            f.write(f"{micrograph}\t{magnification:.6f}\t{pixel_size:.6f}\t{group_number}\t")
            f.write(f"{rot:.6f}\t{tilt:.6f}\t{psi:.6f}\t")
            f.write(f"{origin_x:.6f}\t{origin_y:.6f}\t{origin_z:.6f}\n")


def convert_batch(input_dir, pattern, output_path, pixel_size=7.84, bin_size=4,
                  magnification=10000, workers=10, ctf_template=None):
    """
    Convert multiple XML files to a single STAR file.
    """
    input_path = Path(input_dir)
    
    # Find all matching XML files
    if pattern:
        xml_files = sorted(input_path.glob(pattern))
    else:
        xml_files = sorted(input_path.glob("*.xml"))
    
    if not xml_files:
        print(f"Error: No files matching '{pattern}' found in {input_dir}")
        sys.exit(1)
    
    total_files = len(xml_files)
    print(f"Found {total_files} XML files to process")
    print(f"Using {workers} workers...\n")
    
    # Prepare arguments for parallel processing
    args_list = [(str(f), i+1, total_files) for i, f in enumerate(xml_files)]
    
    # Process XML files in parallel
    all_particles = []
    with Pool(processes=workers) as pool:
        results = pool.map(process_single_xml, args_list)
    
    # Combine results
    for particles in results:
        all_particles.extend(particles)
    
    print(f"\nTotal particles collected: {len(all_particles)}")
    
    if not all_particles:
        print("Error: No particles found")
        sys.exit(1)
    
    # Write STAR file
    print(f"\nWriting STAR file: {output_path}")
    write_star_file(
        all_particles,
        output_path,
        pixel_size=pixel_size,
        bin_size=bin_size,
        magnification=magnification,
        ctf_template=ctf_template
    )
    
    print(f"STAR file written successfully!")
    print(f"  Path: {output_path}")
    print(f"  Particles: {len(all_particles)}")
    print(f"  From {total_files} tomograms")
    
    return len(all_particles)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTom XML particle lists to RELION STAR format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch conversion with parallel processing
  python convert_pytom_to_star.py \\
      --input-dir atp/04_Particle_Picking/Picked_Particles \\
      --pattern "*_atps35A.xml" \\
      --output atp/atps35A_particles.star \\
      --bin-size 4 \\
      --workers 10

  # Single file conversion
  python convert_pytom_to_star.py \\
      --input particles.xml \\
      --output particles.star \\
      --bin-size 4
        """
    )
    
    parser.add_argument('--input', type=str, help='Single input XML file')
    parser.add_argument('--input-dir', type=str, help='Directory containing XML files')
    parser.add_argument('--pattern', type=str, default='*.xml',
                        help='File pattern for batch conversion (default: *.xml)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output STAR file path')
    parser.add_argument('--pixel-size', type=float, default=7.84,
                        help='Detector pixel size in Angstroms (default: 7.84)')
    parser.add_argument('--bin-size', type=float, default=4,
                        help='Binning factor used in PyTom (default: 4)')
    parser.add_argument('--magnification', type=float, default=10000,
                        help='Magnification (default: 10000)')
    parser.add_argument('--ctf-template', type=str, default=None,
                        help='CTF image path template (use {ts_name} or {i} as placeholders)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers for batch conversion (default: 10)')
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error('Either --input or --input-dir must be specified')
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.input:
        # Single file conversion
        print(f"Converting single file: {args.input}")
        particles = parse_xml_particle_list(args.input)
        print(f"Found {len(particles)} particles")
        
        write_star_file(
            particles,
            str(output_path),
            pixel_size=args.pixel_size,
            bin_size=args.bin_size,
            magnification=args.magnification,
            ctf_template=args.ctf_template
        )
        print(f"Output written to: {output_path}")
    else:
        # Batch conversion
        convert_batch(
            input_dir=args.input_dir,
            pattern=args.pattern,
            output_path=str(output_path),
            pixel_size=args.pixel_size,
            bin_size=args.bin_size,
            magnification=args.magnification,
            workers=args.workers,
            ctf_template=args.ctf_template
        )


if __name__ == '__main__':
    main()
