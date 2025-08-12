"""Split EXTXYZ datasets into N chunk files for parallel processing.

This utility reads one or more `.xyz` files (EXTXYZ format), loads all frames
into memory, partitions them into approximately equal-sized chunks, and writes
each chunk to a new `.xyz` file under the specified output directory.

All frame-level metadata in `Atoms.info` and arrays are preserved. The fields
`source_path` and `frame_index` are retained if present; if missing, they are
set to the origin file and per-file index during load.

Examples:
    Split the three files in a data directory into 12 chunks:
        python split_xyz_chunks.py \
            --data-dir /abs/path/to/data \
            --num-splits 12 \
            --output-dir /abs/path/to/output/splits_12

    Or provide explicit files:
        python split_xyz_chunks.py --xyz /f/train.xyz /f/val.xyz /f/test.xyz \
            --num-splits 12 --output-dir /out/splits_12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from ase import Atoms
from ase.io import iread, write


def _collect_frames(xyz_files: Sequence[Path]) -> List[Tuple[Atoms, Path]]:
    """Load all frames from the given `.xyz` files into memory.

    Args:
        xyz_files: Paths to EXTXYZ files.

    Returns:
        List of `(atoms, origin_path)` tuples in the order encountered.

    Raises:
        FileNotFoundError: If any file does not exist.
    """
    results: List[Tuple[Atoms, Path]] = []
    for file_path in xyz_files:
        file_path = file_path.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        for idx, atoms in enumerate(iread(str(file_path), format="extxyz", index=":")):
            # Preserve provenance if not set
            atoms.info.setdefault("source_path", str(file_path))
            atoms.info.setdefault("frame_index", idx)
            results.append((atoms, file_path))
    return results


def _split_evenly(frames: Sequence[Tuple[Atoms, Path]], num_splits: int) -> List[List[Tuple[Atoms, Path]]]:
    """Partition frames into `num_splits` approximately equal chunks.

    Args:
        frames: Sequence of `(Atoms, origin_path)` pairs.
        num_splits: Number of chunks to create.

    Returns:
        List of chunks, each a list of frame tuples.
    """
    total = len(frames)
    if num_splits <= 0:
        raise ValueError("num_splits must be > 0")
    if total == 0:
        return [[] for _ in range(num_splits)]

    base = total // num_splits
    remainder = total % num_splits

    chunks: List[List[Tuple[Atoms, Path]]] = []
    start = 0
    for i in range(num_splits):
        size = base + (1 if i < remainder else 0)
        end = start + size
        chunks.append(list(frames[start:end]))
        start = end
    return chunks


def _write_chunks(chunks: Sequence[Sequence[Tuple[Atoms, Path]]], output_dir: Path) -> List[Path]:
    """Write each chunk to an EXTXYZ file.

    Args:
        chunks: Sequence of frame lists.
        output_dir: Directory to write chunk files.

    Returns:
        Paths to the written chunk files in order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for idx, frames in enumerate(chunks):
        out_path = (output_dir / f"chunk_{idx:04d}.xyz").resolve()
        if out_path.exists():
            out_path.unlink()
        # Write sequentially to preserve order
        for atoms, _origin in frames:
            write(filename=str(out_path), images=atoms, format="extxyz", append=True)
        written.append(out_path)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional sequence of arguments. If None, uses `sys.argv`.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description="Split EXTXYZ datasets into N chunks")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .xyz files; scans recursively for .xyz",
    )
    p.add_argument(
        "--xyz",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit .xyz files (overrides --data-dir if provided)",
    )
    p.add_argument("--num-splits", type=int, required=True, help="Number of chunks to create")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write chunk .xyz files",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for splitting utility.

    This function discovers input files, loads frames, partitions them, and
    writes chunk files into the specified output directory.
    """
    args = parse_args(argv)

    xyz_files: List[Path] = []
    if args.xyz:
        xyz_files = [p.expanduser().resolve() for p in args.xyz]
    elif args.data_dir is not None:
        data_dir = args.data_dir.expanduser().resolve()
        xyz_files = sorted(list(data_dir.rglob("*.xyz")))
    else:
        raise ValueError("Either --xyz or --data-dir must be provided")

    frames = _collect_frames(xyz_files)
    chunks = _split_evenly(frames, args.num_splits)
    written = _write_chunks(chunks, args.output_dir.expanduser().resolve())

    # Print written paths, one per line, for convenience
    for p in written:
        print(p)


if __name__ == "__main__":
    main()


