"""
Extract 3D cubes from a tomogram using particle coordinates in a STAR file.
"""

import argparse
import os

import numpy as np
import starfile

from cryodrgn import mrc


def _norm_name(name: str) -> str:
    return name.strip().lstrip("_").lower()


def _resolve_column(df, requested: str) -> str:
    if requested in df.columns:
        return requested
    lookup = {_norm_name(c): c for c in df.columns}
    key = _norm_name(requested)
    if key in lookup:
        return lookup[key]
    raise KeyError(f"Column not found in STAR: {requested}")


def _load_star_df(path: str, data_block: str = None):
    s = starfile.read(path)
    if isinstance(s, dict):
        if data_block is None:
            if len(s) != 1:
                blocks = ", ".join(list(s.keys()))
                raise ValueError(
                    f"STAR has multiple data blocks ({blocks}); use --data-block"
                )
            return next(iter(s.values()))
        if data_block not in s:
            blocks = ", ".join(list(s.keys()))
            raise KeyError(f"data block {data_block} not found. Available: {blocks}")
        return s[data_block]
    return s


def _round_coords(coords: np.ndarray, mode: str) -> np.ndarray:
    if mode == "round":
        return np.rint(coords).astype(np.int64)
    if mode == "floor":
        return np.floor(coords).astype(np.int64)
    if mode == "ceil":
        return np.ceil(coords).astype(np.int64)
    raise ValueError(f"Unsupported round mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tomo", type=os.path.abspath, help="input tomogram (.mrc/.map)")
    parser.add_argument("star", type=os.path.abspath, help="input STAR file with coordinates")
    parser.add_argument(
        "--coord-cols",
        nargs=3,
        metavar=("XCOL", "YCOL", "ZCOL"),
        default=("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"),
        help="coordinate columns in STAR (default: %(default)s)",
    )
    parser.add_argument(
        "--data-block",
        type=str,
        default=None,
        help="STAR data block name (required only when STAR has multiple blocks)",
    )
    parser.add_argument(
        "--tomo-col",
        type=str,
        default=None,
        help="optional STAR column name used to filter one tomogram",
    )
    parser.add_argument(
        "--tomo-id",
        type=str,
        default=None,
        help="value in --tomo-col to select particles for this tomogram",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        required=True,
        help="cube edge length in voxels",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help="scale factor applied to STAR coordinates before extraction",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="treat input coordinates as 1-based and convert to 0-based",
    )
    parser.add_argument(
        "--round-mode",
        choices=("round", "floor", "ceil"),
        default="round",
        help="how to convert float coordinates to voxel indices",
    )
    parser.add_argument(
        "--pad-outside",
        action="store_true",
        help="pad with zeros when a cube crosses tomogram boundary; default skips",
    )
    parser.add_argument(
        "--out-star",
        type=os.path.abspath,
        default=None,
        help="optional output STAR with only extracted entries",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=os.path.abspath,
        required=True,
        help="output directory for extracted cube files (.mrc)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="cube",
        help="file prefix for extracted cubes (default: %(default)s)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="starting index used in output file names (default: %(default)s)",
    )
    parser.add_argument(
        "--write-stack",
        type=os.path.abspath,
        default=None,
        help="optional output stack path (.mrcs/.mrc) in addition to per-cube files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.box_size <= 0:
        raise ValueError("--box-size must be > 0")
    if args.coord_scale <= 0:
        raise ValueError("--coord-scale must be > 0")

    df = _load_star_df(args.star, args.data_block)

    if args.tomo_col is not None:
        col_tomo = _resolve_column(df, args.tomo_col)
        if args.tomo_id is None:
            raise ValueError("--tomo-id is required when --tomo-col is set")
        df = df[df[col_tomo].astype(str) == str(args.tomo_id)]

    xcol = _resolve_column(df, args.coord_cols[0])
    ycol = _resolve_column(df, args.coord_cols[1])
    zcol = _resolve_column(df, args.coord_cols[2])
    coords_xyz = df[[xcol, ycol, zcol]].to_numpy(dtype=np.float32) * float(args.coord_scale)
    if args.one_based:
        coords_xyz -= 1.0
    idx_xyz = _round_coords(coords_xyz, args.round_mode)

    tomo_lazy, header = mrc.parse_tomo(args.tomo)
    tomo = tomo_lazy.get().astype(np.float32)
    nz, ny, nx = tomo.shape

    b = int(args.box_size)
    h = b // 2
    cubes = []
    keep_indices = []

    for i, (cx, cy, cz) in enumerate(idx_xyz):
        sx, ex = cx - h, cx - h + b
        sy, ey = cy - h, cy - h + b
        sz, ez = cz - h, cz - h + b

        inside = sx >= 0 and sy >= 0 and sz >= 0 and ex <= nx and ey <= ny and ez <= nz
        if inside:
            cube = tomo[sz:ez, sy:ey, sx:ex]
            cubes.append(cube)
            keep_indices.append(i)
            continue

        if not args.pad_outside:
            continue

        cube = np.zeros((b, b, b), dtype=np.float32)
        xs, xe = max(0, sx), min(nx, ex)
        ys, ye = max(0, sy), min(ny, ey)
        zs, ze = max(0, sz), min(nz, ez)
        if xs < xe and ys < ye and zs < ze:
            cube[zs - sz : ze - sz, ys - sy : ye - sy, xs - sx : xe - sx] = tomo[zs:ze, ys:ye, xs:xe]
            cubes.append(cube)
            keep_indices.append(i)

    if len(cubes) == 0:
        raise RuntimeError("No cubes extracted. Check coordinates, box size, and boundary mode.")

    stack = np.stack(cubes, axis=0).astype(np.float32)
    os.makedirs(args.output_dir, exist_ok=True)

    apix = header.get_apix()
    for k, cube in enumerate(stack, start=args.start_index):
        out_cube = os.path.join(args.output_dir, f"{args.prefix}_{k:06d}.mrc")
        mrc.write(out_cube, cube.astype(np.float32), Apix=apix, is_vol=True)

    if args.write_stack is not None:
        os.makedirs(os.path.dirname(args.write_stack) or ".", exist_ok=True)
        mrc.write(args.write_stack, stack, Apix=apix, is_vol=False)

    print(f"Tomogram shape (z,y,x): {tomo.shape}")
    print(f"Input picks: {len(df)}")
    print(f"Extracted cubes: {len(stack)}")
    print(f"Output cube directory: {args.output_dir}")
    if args.write_stack is not None:
        print(f"Output stack: {args.write_stack}")

    if args.out_star is not None:
        os.makedirs(os.path.dirname(args.out_star) or ".", exist_ok=True)
        df_out = df.iloc[np.array(keep_indices, dtype=np.int64)].reset_index(drop=True)
        starfile.write(df_out, args.out_star)
        print(f"Output STAR (extracted picks only): {args.out_star}")


if __name__ == "__main__":
    main()
