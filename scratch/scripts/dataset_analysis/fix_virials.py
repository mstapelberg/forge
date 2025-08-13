#!/usr/bin/env python3
# remove_extxyz_virial.py
# Remove the entire top-level `virial=...` token from extended XYZ headers.
# Prints: <path>\t<structure_id> for each frame modified.

import os
import argparse
import gzip

def split_top_level_tokens(s: str):
    """Split header into tokens on whitespace, but only at top level
    (i.e., not inside quotes or (), [], {})."""
    toks, buf = [], []
    q = None
    esc = False
    d_par, d_brk, d_cur = 0, 0, 0
    for ch in s:
        if esc:
            buf.append(ch); esc = False; continue
        if q:
            if ch == '\\':
                buf.append(ch); esc = True
            elif ch == q:
                buf.append(ch); q = None
            else:
                buf.append(ch)
            continue
        if ch in ('"', "'"):
            q = ch; buf.append(ch); continue
        if ch == '(':
            d_par += 1; buf.append(ch); continue
        if ch == ')':
            d_par = max(0, d_par - 1); buf.append(ch); continue
        if ch == '[':
            d_brk += 1; buf.append(ch); continue
        if ch == ']':
            d_brk = max(0, d_brk - 1); buf.append(ch); continue
        if ch == '{':
            d_cur += 1; buf.append(ch); continue
        if ch == '}':
            d_cur = max(0, d_cur - 1); buf.append(ch); continue
        if ch.isspace() and d_par == d_brk == d_cur == 0:
            if buf:
                toks.append(''.join(buf)); buf = []
            continue
        buf.append(ch)
    if buf:
        toks.append(''.join(buf))
    return toks

def strip_wrapping(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    if len(v) >= 2 and ((v[0], v[-1]) in [('(', ')'), ('[', ']'), ('{', '}')]):
        return v[1:-1]
    return v

def process_header(header: str):
    """Return (new_header, changed, structure_id)."""
    tokens = split_top_level_tokens(header.strip())
    new_tokens = []
    changed = False
    struct_id = None

    for t in tokens:
        if '=' not in t:
            new_tokens.append(t)
            continue
        k, v = t.split('=', 1)
        key = k.strip().lower()

        if key == 'structure_id' and struct_id is None:
            struct_id = strip_wrapping(v)

        if key == 'virial':
            # Drop entire virial=... token
            changed = True
            continue

        new_tokens.append(t)

    new_header = ' '.join(new_tokens)
    return new_header, changed, struct_id

def opener_for(path, mode):
    return gzip.open if path.endswith('.gz') else open

def process_file(path: str, dry_run: bool):
    frames_modified = 0
    file_changed = False

    rd = opener_for(path, 'rt')
    wr = opener_for(path, 'wt') if not dry_run else None
    tmp_path = path + '.tmp'

    with rd(path, 'rt', encoding='utf-8', errors='ignore') as fin:
        fout = None if dry_run else wr(tmp_path, 'wt', encoding='utf-8')
        while True:
            nat = fin.readline()
            if not nat:
                break  # EOF
            if not nat.strip():
                # Preserve stray blanks (rare in extxyz)
                if fout:
                    fout.write(nat)
                continue

            # Parse atom count
            try:
                n = int(nat.strip())
            except ValueError:
                # Not a proper frame start; copy through if writing
                if fout:
                    fout.write(nat)
                continue

            header = fin.readline()
            if not header:
                header = '\n'

            new_header, changed, sid = process_header(header.rstrip('\n'))

            if changed:
                frames_modified += 1
                file_changed = True
                print(f"{path}\t{sid or '<no-structure_id>'}")

            if fout:
                fout.write(f"{n}\n")
                fout.write((new_header if changed else header.rstrip('\n')) + "\n")

            # Copy atom lines verbatim
            for _ in range(n):
                line = fin.readline()
                if fout:
                    fout.write(line)

        if fout:
            fout.close()

    if not dry_run:
        if file_changed:
            os.replace(tmp_path, path)
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return frames_modified, file_changed and not dry_run

def iter_files(root, exts):
    exts = tuple(e.lower() for e in exts)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                yield os.path.join(dp, fn)

def main():
    ap = argparse.ArgumentParser(description="Remove top-level virial=... from extended XYZ headers.")
    ap.add_argument("root", help="Root directory to search")
    ap.add_argument("--dry-run", action="store_true", help="Print what will change; don’t rewrite files")
    ap.add_argument("--ext", action="append", default=[".xyz"], help="Extensions to include (repeatable). Default: .xyz")
    args = ap.parse_args()

    total_files = 0
    total_frames = 0
    total_files_changed = 0

    for path in iter_files(args.root, args.ext):
        total_files += 1
        frames, wrote = process_file(path, args.dry_run)
        if frames > 0:
            total_frames += frames
            total_files_changed += 1

    print(f"# scanned_files={total_files}  modified_files={total_files_changed}  frames_updated={total_frames}")

if __name__ == "__main__":
    main()

