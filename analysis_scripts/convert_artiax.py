"""
Convert a STAR file to an ArtiaX-compatible STAR schema.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import starfile


ARTIAX_COLUMNS = [
    "rlnTomoName",
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnOriginX",
    "rlnOriginY",
    "rlnOriginZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
]


def _read_star_df(path: str, data_block: str = None) -> pd.DataFrame:
    data = starfile.read(path)
    if isinstance(data, dict):
        if data_block is None:
            if len(data) != 1:
                blocks = ", ".join(data.keys())
                raise ValueError(
                    f"Multiple data blocks found ({blocks}); use --data-block"
                )
            return next(iter(data.values()))
        if data_block not in data:
            blocks = ", ".join(data.keys())
            raise KeyError(f"data block {data_block} not found. Available: {blocks}")
        return data[data_block]
    return data


def convert_star_to_artiax(
    df: pd.DataFrame,
    mic: str,
    factor: float = 2.132,
    deduplicate: bool = False,
    micrograph_col: str = "rlnMicrographName",
) -> pd.DataFrame:
    if micrograph_col not in df.columns:
        raise KeyError(f"Column not found: {micrograph_col}")

    coord_cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    for c in coord_cols:
        if c not in df.columns:
            raise KeyError(f"Required coordinate column missing: {c}")

    # Filter by micrograph name, same convention as the user's script.
    df = df[df[micrograph_col].astype(str) == f"{mic}.tomostar"].copy()

    if deduplicate:
        if "rlnImageName" not in df.columns:
            raise KeyError("deduplicate requires column rlnImageName")
        df = df.groupby("rlnImageName", sort=False).first().reset_index()

    # Scale coordinates.
    df["rlnCoordinateX"] = df["rlnCoordinateX"].astype(float) * factor
    df["rlnCoordinateY"] = df["rlnCoordinateY"].astype(float) * factor
    df["rlnCoordinateZ"] = df["rlnCoordinateZ"].astype(float) * factor

    # Fill required ArtiaX columns.
    out = pd.DataFrame(index=df.index)
    out["rlnTomoName"] = mic
    out["rlnCoordinateX"] = df["rlnCoordinateX"]
    out["rlnCoordinateY"] = df["rlnCoordinateY"]
    out["rlnCoordinateZ"] = df["rlnCoordinateZ"]

    for col in ["rlnOriginX", "rlnOriginY", "rlnOriginZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]:
        if col in df.columns:
            out[col] = df[col].astype(float)
        else:
            # Keep schema complete for ArtiaX compatibility.
            out[col] = 0.0

    return out[ARTIAX_COLUMNS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=os.path.abspath, help="input STAR file")
    parser.add_argument("mic", type=str, help="tomogram/micrograph identifier")
    parser.add_argument(
        "--factor",
        type=float,
        default=2.132,
        help="coordinate scaling factor (default: %(default)s)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="remove duplicates using groupby('rlnImageName').first()",
    )
    parser.add_argument(
        "--micrograph-col",
        type=str,
        default="rlnMicrographName",
        help="column used to filter micrograph (default: %(default)s)",
    )
    parser.add_argument(
        "--data-block",
        type=str,
        default=None,
        help="STAR data block name for multi-block STAR files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=os.path.abspath,
        default=None,
        help="output STAR file path (default: <input_stem>_<mic>.star)",
    )
    return parser


def main(args):
    df = _read_star_df(args.input, args.data_block)
    out_df = convert_star_to_artiax(
        df,
        mic=args.mic,
        factor=args.factor,
        deduplicate=args.deduplicate,
        micrograph_col=args.micrograph_col,
    )

    if args.output is None:
        output_path = f"{Path(args.input).stem}_{args.mic}.star"
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    starfile.write(out_df, output_path)
    print(f"Saved {len(out_df)} particles to {output_path}")


if __name__ == "__main__":
    main(parse_args().parse_args())
