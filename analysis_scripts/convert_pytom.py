import starfile
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tempfile


def _sanitize_pytom_loop_header_spacing(text: str) -> str:
    """Remove blank lines between `loop_` and first `_rln...` header."""
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.strip() != "loop_":
            i += 1
            continue

        j = i + 1
        blank_lines = []
        while j < len(lines) and lines[j].strip() == "":
            blank_lines.append(lines[j])
            j += 1

        # PyTOM export sometimes inserts an empty line here:
        # loop_
        #
        # _rlnXXX ...
        if j < len(lines) and lines[j].lstrip().startswith("_rln"):
            # Drop the blank lines in this specific case.
            i = j
            continue

        # Otherwise keep original content.
        out.extend(blank_lines)
        i = j

    return "".join(out)


def read_starfile_with_pytom_fix(file_path: str):
    text = Path(file_path).read_text()
    cleaned = _sanitize_pytom_loop_header_spacing(text)
    if cleaned == text:
        return starfile.read(file_path)

    print("Detected blank line(s) after loop_; applying PyTOM STAR spacing fix.")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".star", delete=False
        ) as tmp_f:
            tmp_f.write(cleaned)
            tmp_path = tmp_f.name
        return starfile.read(tmp_path)
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

file_name = sys.argv[1]
assert file_name.endswith('.star')
df = read_starfile_with_pytom_fix(file_name)
mic = sys.argv[2]
df = df[df['rlnMicrographName'].str.contains(mic)] #for retrieving template matching
#df = df[df['rlnMicrographName'] == (mic+".tomostar")] #for exporting m
print(len(df), mic)
df['rlnMicrographName'] = f"{mic}.tomostar"
coords = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
print(f'export the starfile for {mic} to {mic}_norm.star')
starfile.write(df[['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                   'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
                   ]], mic+'_norm.star')
sys.exit(0)
