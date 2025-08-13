"""Compare structure IDs between two Allegro job directories.

This utility reads `structure_splits.json` from each provided job directory and
compares the union of train/val/test IDs. It prints a short report of:

- Count of IDs per job
- Number and list of IDs only in A
- Number and list of IDs only in B
- Unique exclusion reasons (from a reasons JSON) for IDs only in A or only in B

Usage:
    python compare_job_structure_ids.py /path/to/job_A /path/to/job_B [path/to/filtered_out_ids_with_reasons.json]

If the third argument is omitted, the script defaults to
`scratch/scripts/dataset_analysis/filtered_out_ids_with_reasons.json`.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _load_split_ids(job_dir: Path) -> Set[int]:
    splits_path = job_dir / 'structure_splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing structure_splits.json in {job_dir}")
    data: Dict[str, List[int]] = json.loads(splits_path.read_text())
    all_ids: Set[int] = set()
    for split_name in ('train', 'val', 'test'):
        all_ids.update(int(x) for x in data.get(split_name, []))
    return all_ids

def _load_reason_map(reasons_path: Path) -> Dict[int, Set[str]]:
    """Load a mapping from structure_id to a set of exclusion reasons.

    Args:
        reasons_path: Path to JSON with keys "excluded": [{"structure_id": int, "reasons": [str, ...]}].

    Returns:
        Dict mapping structure_id to a set of reasons.
    """
    reason_map: Dict[int, Set[str]] = {}
    if not reasons_path.exists():
        return reason_map
    payload = json.loads(reasons_path.read_text())
    for entry in payload.get("excluded", []):
        try:
            sid = int(entry.get("structure_id"))
            reasons = entry.get("reasons", []) or []
            reason_map[sid] = set(str(r) for r in reasons)
        except Exception:
            continue
    return reason_map


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print("Usage: python compare_job_structure_ids.py /path/to/job_A /path/to/job_B [path/to/filtered_out_ids_with_reasons.json]")
        sys.exit(1)
    job_a = Path(sys.argv[1])
    job_b = Path(sys.argv[2])
    default_reasons = Path('dataset_analysis/filtered_out_ids_with_reasons.json')
    reasons_path = Path(sys.argv[3]) if len(sys.argv) == 4 else default_reasons

    ids_a = _load_split_ids(job_a)
    ids_b = _load_split_ids(job_b)

    only_a = sorted(list(ids_a - ids_b))
    only_b = sorted(list(ids_b - ids_a))

    print(f"Job A: {job_a}")
    print(f"Job B: {job_b}")
    print(f"Total IDs A: {len(ids_a)} | Total IDs B: {len(ids_b)}")
    print(f"Only in A ({len(only_a)}): {only_a[:50]}{' ...' if len(only_a) > 50 else ''}")
    print(f"Only in B ({len(only_b)}): {only_b[:50]}{' ...' if len(only_b) > 50 else ''}")

    # Load exclusion reasons and report unique reasons for unmatched IDs
    reason_map = _load_reason_map(reasons_path)
    if reason_map:
        unique_reasons_a = sorted(list({r for sid in only_a for r in reason_map.get(sid, set())}))
        unique_reasons_b = sorted(list({r for sid in only_b for r in reason_map.get(sid, set())}))
        print(f"Unique reasons for IDs only in A ({len(unique_reasons_a)}): {unique_reasons_a}")
        print(f"Unique reasons for IDs only in B ({len(unique_reasons_b)}): {unique_reasons_b}")
    else:
        print(f"No reasons file found or could not parse reasons at: {reasons_path}")


if __name__ == '__main__':
    main()


