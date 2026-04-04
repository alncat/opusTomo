import argparse

import starfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert STAR file and optionally export a specific random subset."
    )
    parser.add_argument("starfile", help="input STAR file")
    parser.add_argument("angpix", type=float, help="angstrom per pixel")
    parser.add_argument(
        "--rescale-angpix",
        type=float,
        default=None,
        help="target angpix for rescaling pixel-based coordinates/translations",
    )
    parser.add_argument(
        "--subset-label",
        type=int,
        default=None,
        help="when set, also write out only this _rlnRandomSubset label",
    )
    parser.add_argument(
        "--remove-symexp",
        action="store_true",
        help="remove symmetry expansion by grouping on rlnImageName",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_name = args.starfile
    if not file_name.endswith(".star"):
        raise ValueError("input file must end with .star")

    df = starfile.read(file_name)
    origins = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    origins_new = ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]

    if origins[0] in df.columns:
        df[origins_new] = df[origins] / args.angpix
        df.drop(origins, axis=1, inplace=True)

    if args.rescale_angpix is not None:
        if args.rescale_angpix <= 0:
            raise ValueError("--rescale-angpix must be > 0")
        scale = float(args.angpix) / float(args.rescale_angpix)
        # Rescale pixel-space coordinates/translations to the new angpix.
        px_cols = [
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnOriginX",
            "rlnOriginY",
            "rlnOriginZ",
        ]
        for col in px_cols:
            if col in df.columns:
                df[col] = df[col] * scale
        # Update common STAR pixel-size metadata when present.
        apix_cols = ["rlnImagePixelSize", "rlnMicrographPixelSize", "rlnPixelSize"]
        for col in apix_cols:
            if col in df.columns:
                df[col] = float(args.rescale_angpix)
        print(
            f"rescale angpix: {args.angpix} -> {args.rescale_angpix}, "
            f"pixel scale factor={scale:.6g}"
        )

    if args.remove_symexp:
        if "rlnImageName" not in df.columns:
            raise KeyError("rlnImageName is required for --remove-symexp")
        n_before = len(df)
        # Remove symmetry expansion by keeping one row per input image.
        df = df.groupby("rlnImageName", sort=False).first().reset_index()
        print(f"remove symmetry expansion: {n_before} -> {len(df)} particles")

    df["rlnRandomSubset"] = (df.index) % 2 + 1
    out_all = file_name[:-5] + "30.star"
    print(f"write out {len(df)} particles with even/odd split to {out_all}")
    starfile.write(df, out_all)

    if args.subset_label is not None:
        df_subset = df[df["rlnRandomSubset"] == args.subset_label]
        if len(df_subset) == 0:
            raise ValueError(f"No particles found for rlnRandomSubset={args.subset_label}")
        subset_out = f"{file_name[:-5]}30_subset{args.subset_label}.star"
        print(
            f"write out {len(df_subset)} particles for randomsubset={args.subset_label} to {subset_out}"
        )
        starfile.write(df_subset, subset_out)


if __name__ == "__main__":
    main()
