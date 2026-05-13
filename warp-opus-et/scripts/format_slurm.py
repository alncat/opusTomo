#!/usr/bin/env python3
"""
Format all .slurm scripts for improved readability.

Transformations (none change any logic or variable values):
1. Align inline comments in CONFIGURATION variable assignments
2. Consistent blank-line spacing (no double blanks, one before echo "Setting up")
3. Normalize sub-section comment headers to # --- Text ---
4. Fix 4-space indentation in if/fi, for/do/done blocks (protects heredocs)
5. Ensure source ~/.bashrc + conda activate + ulimit -v unlimited order
6. echo ==== banners use consistent 60-char width
"""

import re
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()


def write_file(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# 6. Echo banner width to 60 chars
# ---------------------------------------------------------------------------
def fix_banners(lines):
    """Change echo "===...===" banners to exactly 60 = signs."""
    result = []
    for line in lines:
        stripped = line.rstrip('\n')
        m = re.match(r'^(echo\s+")(=+)(")$', stripped)
        if m and len(m.group(2)) != 60:
            result.append(m.group(1) + '=' * 60 + m.group(3) + '\n')
        else:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# 2/3. Blank-line consistency
# ---------------------------------------------------------------------------
def normalize_blank_lines(lines):
    """Collapse duplicate blank lines and ensure spacing."""
    result = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        prev_blank = is_blank
        result.append(line)

    return result


# ---------------------------------------------------------------------------
# Config block detection
# ---------------------------------------------------------------------------
def find_config_blocks(lines):
    """Return list of (start, end) line indices for CONFIGURATION blocks."""
    blocks = []
    in_config = False
    start = 0

    config_header_re = re.compile(r'^\s*#\s*-{3,}\s*CONFIGURATION')
    section_end_re = re.compile(
        r'^\s*#\s*-{3,}\s+(Setup\s+Environment|Environment Setup|Setup\b'
        r'|Find|Process|Convert|Generate|Build|Export|Import|Verify'
        r'|Run\b|Create|Summary|Check|Calculate|Step|Read|Build MCore'
        r'|Run Training|Generate|Build cmd|Run\s+Parallel|Environment\b'
        r'|Create species)'
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_config and config_header_re.match(stripped):
            in_config = True
            start = i
        elif in_config:
            if re.match(r'^echo\s+"Setting up', stripped):
                blocks.append((start, i))
                in_config = False
            elif section_end_re.match(stripped):
                blocks.append((start, i))
                in_config = False
    if in_config:
        blocks.append((start, len(lines)))
    return blocks


# ---------------------------------------------------------------------------
# 1. Align inline comments in CONFIGURATION variable assignments
# ---------------------------------------------------------------------------
def align_config_comments(lines, blocks):
    """Align inline # comments after VAR=value in config blocks."""
    result = list(lines)

    for block_start, block_end in blocks:
        var_entries = []

        for i in range(block_start, block_end):
            raw = result[i].rstrip('\n')

            patterns = [
                r'^([A-Za-z_][A-Za-z0-9_]*="(?:[^"]*)")\s+(#.*)$',
                r'^([A-Za-z_][A-Za-z0-9_]*=[0-9.eE+\-]+)\s+(#.*)$',
                r'^([A-Za-z_][A-Za-z0-9_]*=[0-9])\s+(#.*)$',
                r'^([A-Za-z_][A-Za-z0-9_]*="")\s+(#.*)$',
                r"^([A-Za-z_][A-Za-z0-9_]*='[^']*')\s+(#.*)$",
                r'^([A-Za-z_][A-Za-z0-9_]*=[A-Z_][A-Z_0-9]*)\s+(#.*)$',
            ]
            for pat in patterns:
                m = re.match(pat, raw)
                if m:
                    var_entries.append((i, m.group(1), m.group(2)))
                    break

        if not var_entries:
            continue

        max_len = max(len(e[1]) for e in var_entries)

        for idx, var_part, comment in var_entries:
            pad = max_len - len(var_part) + 4
            result[idx] = var_part + ' ' * pad + comment.rstrip() + '\n'

    return result


# ---------------------------------------------------------------------------
# 3. Normalize sub-section headers
# ---------------------------------------------------------------------------
def normalize_headers(lines, blocks):
    """Convert plain # comment section headers in CONFIG to # --- Text ---."""
    result = list(lines)
    heading_re = re.compile(r'^#\s+([A-Za-z].*)$')

    for block_start, block_end in blocks:
        i = block_start
        while i < block_end:
            raw = result[i].rstrip('\n')
            m = heading_re.match(raw)
            if m:
                text = m.group(1).strip()
                if '---' in raw or '===' in raw or text.endswith('.'):
                    i += 1
                    continue

                j = i + 1
                while j < block_end and result[j].strip() == '':
                    j += 1

                if j < block_end:
                    nxt = result[j].strip()
                    if nxt.startswith('#'):
                        i += 1
                        continue
                    if (re.match(r'^[A-Za-z_]+\s*=', nxt)
                            or nxt.startswith('echo ')
                            or nxt.startswith('export ')
                            or nxt.startswith('case ')):
                        if len(text) < 70:
                            new_line = '# --- ' + text + ' ---\n'
                            if new_line != result[i]:
                                result[i] = new_line
            i += 1

    return result


# ---------------------------------------------------------------------------
# 5. source/conda/ulimit ordering
# ---------------------------------------------------------------------------
def fix_env_ordering(lines):
    """Move `ulimit -v unlimited` that appears before `source ~/.bashrc`
    to after the `conda activate` block. Also remove blank lines between
    source, conda activate, and ulimit when they appear consecutively."""
    result = list(lines)

    # Pass 1: find ulimit before source and reorder
    for i in range(len(result)):
        if result[i].strip() == 'ulimit -v unlimited':
            # Check if source ~/.bashrc appears within 15 lines AFTER this ulimit
            source_idx = None
            for j in range(i + 1, min(i + 15, len(result))):
                if 'source ~/.bashrc' in result[j]:
                    source_idx = j
                    break

            if source_idx is not None:
                # ulimit is before source -- find the conda activate block
                conda_end = source_idx
                for j in range(source_idx + 1, min(source_idx + 15, len(result))):
                    s = result[j].strip()
                    if s.startswith('if ! conda activate') or s.startswith('conda activate'):
                        k = j
                        if s.startswith('if ! conda activate'):
                            # Find the matching fi
                            depth = 1
                            k += 1
                            while k < len(result) and depth > 0:
                                ks = result[k].strip()
                                if ks.startswith('if '):
                                    depth += 1
                                elif ks == 'fi':
                                    depth -= 1
                                k += 1
                        else:
                            k += 1
                        conda_end = k
                        break

                if conda_end > source_idx:
                    # Move ulimit line to after conda_end
                    ulimit_line = result.pop(i)
                    if conda_end > i:
                        conda_end -= 1
                    result.insert(conda_end, ulimit_line)
                break  # only one block to fix

    # Pass 2: remove blank lines between source, conda activate, and ulimit
    # when they appear consecutively (source -> conda -> ulimit)
    result2 = []
    i = 0
    while i < len(result):
        stripped = result[i].strip()
        if 'source ~/.bashrc' in stripped:
            result2.append(result[i])
            i += 1
            # Skip blanks to find conda activate
            blanks = 0
            while i < len(result) and result[i].strip() == '':
                blanks += 1
                i += 1
            if i < len(result) and ('conda activate' in result[i]
                                     or result[i].strip().startswith('if ! conda activate')):
                # Check if there are non-blank lines between source and conda
                # If so, these aren't consecutive -- preserve original spacing
                if blanks > 0 and result2[-1].strip() != result[i].strip():
                    # There were non-blank (comment) lines in between
                    for _ in range(blanks):
                        result2.append('\n')
                result2.append(result[i])
                i += 1
                # After conda activate, skip blanks to find ulimit
                if result[i-1].strip().startswith('if ! conda activate'):
                    # Read until fi
                    while i < len(result) and result[i].strip() != 'fi':
                        result2.append(result[i])
                        i += 1
                    if i < len(result) and result[i].strip() == 'fi':
                        result2.append(result[i])
                        i += 1
                blanks2 = 0
                while i < len(result) and result[i].strip() == '':
                    blanks2 += 1
                    i += 1
                if i < len(result) and 'ulimit -v unlimited' in result[i].strip():
                    result2.append(result[i])
                    i += 1
                else:
                    # No ulimit, restore blanks
                    for _ in range(blanks2):
                        result2.append('\n')
            else:
                # No conda activate after source, restore blanks
                for _ in range(blanks):
                    result2.append('\n')
        else:
            result2.append(result[i])
            i += 1

    return result2


# ---------------------------------------------------------------------------
# 4. Fix indentation in if/fi, for/do/done blocks
# ---------------------------------------------------------------------------
def fix_indentation(lines):
    """Ensure 4-space indentation. Protects heredocs and SBATCH directives."""
    result = []
    heredoc_delim = None
    indent = 0

    for line in lines:
        # Heredoc content: pass through with original indentation
        if heredoc_delim is not None:
            if line.strip() == heredoc_delim:
                heredoc_delim = None
            result.append(line)
            continue

        # Heredoc start detection
        m = re.search(r'<<\s*(\'?"?(\w+)"?\'?)', line)
        if m:
            heredoc_delim = m.group(2)
            result.append(line)
            continue

        stripped = line.lstrip()
        content = stripped.rstrip('\n')
        leading = len(line) - len(stripped)

        # Skip blank / comment / SBATCH lines -- preserve original indent
        if not content or content.startswith('#'):
            result.append(line)
            continue

        first_word = content.split()[0] if content.split() else ''

        # ---- Determine indent BEFORE writing this line ----

        if first_word in ('fi', 'done', 'esac'):
            indent -= 4
        elif first_word == 'else':
            indent -= 4
        elif content.startswith('elif '):
            indent -= 4

        indent = max(0, indent)

        # ---- Write the line with current indent ----
        new_indent = max(0, indent)
        if leading != new_indent:
            result.append(' ' * new_indent + content + '\n')
        else:
            result.append(line)

        # ---- Determine indent AFTER writing ----

        # Detect one-liner (if/fi or for/done on same line) -- no indent change
        words = set(content.split())
        is_if_fi_one_liner = content.startswith('if ') and 'fi' in words
        is_for_done_one_liner = content.startswith('for ') and 'done' in words

        if not (is_if_fi_one_liner or is_for_done_one_liner):
            if first_word == 'then':
                indent += 4
            elif first_word == 'do':
                indent += 4
            elif first_word == 'else':
                indent += 4
            elif content.startswith('elif '):
                if '; then' in content:
                    indent += 4
            elif (content.startswith('if ') or content.startswith('for ')
                  or content.startswith('while ') or content.startswith('case ')):
                if '; then' in content or '; do' in content:
                    indent += 4

    return result


# ---------------------------------------------------------------------------
# Final cleanup: ensure one blank line before echo "Setting up"
# ---------------------------------------------------------------------------
def final_cleanup(lines):
    """Ensure blank line before 'echo Setting up', remove double blanks."""
    result = []
    prev_blank = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^echo\s+["\']Setting up', stripped):
            if i > 0 and lines[i - 1].strip() != '':
                # Add blank line before
                result.append('\n')
        result.append(line)

    # Remove any double blanks
    result2 = []
    prev_blank = False
    for line in result:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        prev_blank = is_blank
        result2.append(line)

    return result2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_file(filepath):
    print(f"  {os.path.basename(filepath)}")
    lines = read_file(filepath)
    original = lines.copy()

    # 6. Fix banner widths
    lines = fix_banners(lines)

    # 2/3. Normalize blank lines
    lines = normalize_blank_lines(lines)

    # 1. Find CONFIG blocks and align comments (first pass)
    blocks = find_config_blocks(lines)
    lines = align_config_comments(lines, blocks)

    # 3. Normalize sub-section headers
    lines = normalize_headers(lines, blocks)

    # 5. Fix source/conda/ulimit ordering
    lines = fix_env_ordering(lines)

    # 4. Fix indentation
    lines = fix_indentation(lines)

    # Final cleanup
    lines = final_cleanup(lines)

    # Re-run comment alignment (in case spacing changed)
    blocks = find_config_blocks(lines)
    lines = align_config_comments(lines, blocks)

    if lines != original:
        write_file(filepath, lines)
        return True
    return False


def main():
    slurm_files = sorted([
        f for f in os.listdir(SCRIPTS_DIR)
        if f.endswith('.slurm')
    ])

    print(f"Formatting {len(slurm_files)} .slurm files in {SCRIPTS_DIR}\n")
    modified = []
    for fname in slurm_files:
        filepath = os.path.join(SCRIPTS_DIR, fname)
        if process_file(filepath):
            modified.append(fname)

    print(f"\nModified: {len(modified)}/{len(slurm_files)} files")
    for m in modified:
        print(f"  - {m}")


if __name__ == '__main__':
    main()
