"""
Hybrid MD+MC WC-parameter sweep over supercell sizes.

This script builds supercells over a size range, runs the hybrid
MCMC-MD sampler, and records Warren–Cowley short-range order (SRO)
after each MD step. Results are saved to CSV and plotted per size.

Usage examples:
    # Build V bcc primitive (cubic=False), substitute Cr/Ti/W/Zr at given at% (floored),
    # sweep sizes 3..12, 15 cycles, 1000 MD steps/cycle, MC steps = N_atoms
    python scratch/scripts/hybrid_mdmc_wc_sweep.py \
        --model /path/to/allegro_model.pth \
        --size-min 3 --size-max 12 \
        --cycles 15 --md-steps 1000 --mc-steps natoms \
        --device cuda --outdir ./wc_sweep_out \
        --seed 42

    # Alternatively, provide your own structure instead of building bcc V:
    python scratch/scripts/hybrid_mdmc_wc_sweep.py \
        --model /path/to/allegro_model.pth \
        --structure /path/to/structure.ext \
        --size-min 3 --size-max 12

Notes:
- The script expects a periodic input `ase.Atoms` structure. It will repeat it
  by (n, n, n) for n in [size_min, size_max]. The `cubic` option is not required
  when using an explicit input structure.
- Warren–Cowley shells are heuristically inferred; by default the first shell is
  set to 1.2 × the minimum non-zero neighbor distance within half the shortest
  cell length. Additional shells are geometric expansions of this bound. You can
  change the number of shells with `--num-shells`.

Confidence: 8/10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.build import bulk
from ase.neighborlist import neighbor_list

from forge.analysis.wc_sro import WarrenCowleyCalculator
from forge.calculators.factory import create_ensemble_calculator
from forge.workflows.hybrid_mcmc import HybridMCMCSampler


class WCSamplingHybridSampler(HybridMCMCSampler):
    """Hybrid sampler that records a scalar WC metric after each MD step.

    This subclass overrides the MD phase to compute a scalar Warren–Cowley
    metric after each MD integration step and appends it to `wc_series`.

    Args:
        wc_shells: Shell boundaries for WC calculator, e.g., [0.0, r1] for
            first shell only or [0.0, r1, r2, ...] for multiple shells.
        species_pair: Optional species pair (e.g. ("Cu", "Ni")) to extract
            WC_{i,j} for that pair. If not provided, uses the mean absolute
            off-diagonal of the first available shell.
        wc_shell_index: Which shell index to report (default: 0 for first).
        aggregate: Aggregation for WC matrix when `species_pair` is None.
            Options: "mean_abs_offdiag" (default), "mean_abs_all",
            "fro_norm" (Frobenius norm), "trace".
        All other args are passed to the base `HybridMCMCSampler`.
    """

    def __init__(
        self,
        *args,
        wc_shells: Sequence[float],
        species_pair: Optional[Tuple[str, str]] = None,
        wc_shell_index: int = 0,
        aggregate: str = "mean_abs_offdiag",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wc_shells: List[float] = list(wc_shells)
        self.wc_shell_index: int = wc_shell_index
        self.aggregate: str = aggregate
        self.species_pair: Optional[Tuple[str, str]] = species_pair
        self.wc_series: List[float] = []

    def _run_md_phase(self, cycle: int) -> None:
        """Run MD, recording WC after each single MD step."""
        # Based on the base implementation, but insert WC capture per step
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md import VelocityVerlet, Langevin

        if not hasattr(self.atoms, 'get_velocities') or self.atoms.get_velocities() is None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.md_temperature)

        if self.md_thermostat == 'langevin':
            md = Langevin(self.atoms, self.md_timestep, temperature_K=self.md_temperature, friction=self.friction)
        else:
            md = VelocityVerlet(self.atoms, self.md_timestep)

        # inner loop; always explicit to capture WC per step
        if self.verbose and self.md_steps_per_cycle > 10:
            from tqdm import tqdm

            md_pbar = tqdm(
                range(self.md_steps_per_cycle),
                desc=f"MD Phase (Cycle {cycle})",
                unit="step",
                leave=False,
            )
            for _ in md_pbar:
                md.run(1)
                self._record_wc_scalar()
            md_pbar.close()
        else:
            for _ in range(self.md_steps_per_cycle):
                md.run(1)
                self._record_wc_scalar()

    def _record_wc_scalar(self) -> None:
        """Compute and append the scalar WC metric for the configured shell.

        The method computes WC with the configured `wc_shells`, picks the
        requested shell index, and derives a scalar either for a specific
        species pair or via an aggregation function.
        """
        wc = WarrenCowleyCalculator(self.atoms, list(self.wc_shells))
        params_by_shell: Dict[int, np.ndarray] = wc.calculate_parameters()
        if self.wc_shell_index not in params_by_shell:
            # No neighbors in this shell at this step; append NaN
            self.wc_series.append(np.nan)
            return

        params = params_by_shell[self.wc_shell_index]

        if self.species_pair is not None:
            # Extract index mapping from WC concentrations order
            types = list(wc.concentrations.keys())
            try:
                i = types.index(self.species_pair[0])
                j = types.index(self.species_pair[1])
            except ValueError:
                # Species not present; record NaN
                self.wc_series.append(np.nan)
                return
            value = float(params[i, j])
            self.wc_series.append(value)
            return

        # Aggregations
        if self.aggregate == "mean_abs_offdiag":
            mask = ~np.eye(params.shape[0], dtype=bool)
            vals = np.abs(params[mask])
            self.wc_series.append(float(np.nanmean(vals) if vals.size else np.nan))
        elif self.aggregate == "mean_abs_all":
            self.wc_series.append(float(np.mean(np.abs(params))))
        elif self.aggregate == "fro_norm":
            self.wc_series.append(float(np.linalg.norm(params)))
        elif self.aggregate == "trace":
            self.wc_series.append(float(np.trace(params)))
        else:
            # Fallback to mean_abs_offdiag
            mask = ~np.eye(params.shape[0], dtype=bool)
            vals = np.abs(params[mask])
            self.wc_series.append(float(np.nanmean(vals) if vals.size else np.nan))


def infer_wc_shells(
    atoms: Atoms,
    num_shells: int = 1,
    first_shell_multiplier: float = 1.2,
    shell_growth: float = 1.4,
) -> List[float]:
    """Infer Warren–Cowley shell boundaries heuristically.

    The first shell is estimated from the minimum non-zero neighbor distance
    within a cutoff of half the shortest cell length. Subsequent shells are
    grown geometrically by `shell_growth`.

    Args:
        atoms: Periodic structure with a cell.
        num_shells: Number of shells (1 = first shell only).
        first_shell_multiplier: Multiplier to set the first shell upper bound
            relative to the minimum non-zero neighbor distance.
        shell_growth: Geometric factor for additional shell upper bounds.

    Returns:
        Shell boundary list suitable for `WarrenCowleyCalculator`.
        For N shells returns N+1 boundaries [0, r1, r2, ..., rN].
    """
    cell_lengths = np.asarray(atoms.cell.lengths())
    if not np.all(cell_lengths > 0):
        raise ValueError("Input atoms must be periodic with a valid cell to infer shells.")

    # Cutoff is half the shortest cell length (avoid double counting across images)
    cutoff = 0.5 * float(np.min(cell_lengths)) - 1e-6
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    if d.size == 0:
        # Fallback: set arbitrary small shell; user should override
        return [0.0, 2.0]

    dpos = d[d > 1e-8]
    if dpos.size == 0:
        return [0.0, 2.0]

    dmin = float(np.min(dpos))
    r1 = first_shell_multiplier * dmin
    shells = [0.0, r1]
    while len(shells) - 1 < num_shells:
        shells.append(shells[-1] * shell_growth)
    return shells[: num_shells + 1]


def _apply_random_substitutions(
    atoms: Atoms,
    target_atpercent: Dict[str, float],
    host_symbol: str,
    rng: np.random.Generator,
) -> None:
    """Randomly substitute host atoms to reach floored integer counts by at%.

    Args:
        atoms: Supercell to modify in-place.
        target_atpercent: Map of element -> target at% (e.g., {"Cr": 1.5, ...}).
        host_symbol: The symbol to be replaced (e.g., 'V').
        rng: Numpy Generator for reproducibility.

    Raises:
        ValueError: If target counts exceed available host sites.
    """
    num_atoms = len(atoms)
    # Compute floored counts per solute
    target_counts = {sym: int(np.floor(num_atoms * (pct / 100.0))) for sym, pct in target_atpercent.items()}
    total_to_substitute = int(sum(target_counts.values()))

    # Find indices of host atoms
    host_indices = [idx for idx, sym in enumerate(atoms.get_chemical_symbols()) if sym == host_symbol]
    if total_to_substitute > len(host_indices):
        raise ValueError(
            f"Requested substitutions ({total_to_substitute}) exceed available host atoms ({len(host_indices)})."
        )

    # Randomly choose host indices to substitute
    chosen = rng.choice(host_indices, size=total_to_substitute, replace=False)
    rng.shuffle(chosen)

    # Assign in fixed element order for determinism
    offset = 0
    for sym in sorted(target_counts.keys()):
        count = target_counts[sym]
        for idx in chosen[offset: offset + count]:
            atoms[idx].symbol = sym
        offset += count


def build_calculator(model: str, device: str, backend: str, species_map: Optional[Path], extra_kwargs: Dict) -> object:
    """Create an ASE calculator or ensemble calculator for the model.

    Args:
        model: Path to model file.
        device: 'cpu' or 'cuda'.
        backend: Backend name, e.g. 'allegro' or 'auto'.
        species_map: Optional JSON file mapping element -> type name (Allegro).
        extra_kwargs: Additional backend kwargs.

    Returns:
        ASE-compatible calculator instance.
    """
    kwargs = dict(extra_kwargs)
    if species_map is not None:
        with open(species_map, 'r') as f:
            kwargs.update(json.load(f))
    calc = create_ensemble_calculator(model_paths=model, backend=backend, device=device, **kwargs)
    return calc


def run_for_size(
    base_atoms: Atoms,
    n: int,
    calculator: object,
    cycles: int,
    md_steps: int,
    mc_steps: Optional[int],
    temperature: float,
    md_temperature: Optional[float],
    wc_shells: Sequence[float],
    species_pair: Optional[Tuple[str, str]],
    aggregate: str,
    verbose: bool,
) -> Tuple[List[float], Atoms]:
    """Run the hybrid sampler for a given supercell size and collect WC series.

    Args:
        base_atoms: Input primitive or small cell.
        n: Supercell replication factor (n, n, n).
        calculator: ASE calculator.
        cycles: Number of hybrid cycles (MD+MC).
        md_steps: MD steps per cycle.
        mc_steps: MC steps per cycle; if None, uses number of atoms.
        temperature: MC temperature in K.
        md_temperature: MD thermostat target temperature in K.
        wc_shells: Warren–Cowley shell boundaries.
        species_pair: Optional pair to extract WC_{i,j}.
        aggregate: Aggregation when pair is not provided.
        verbose: Whether to show progress bars.

    Returns:
        (wc_series, final_atoms)
    """
    atoms = base_atoms.copy().repeat((n, n, n))
    atoms.calc = calculator

    num_atoms = len(atoms)
    mc_steps_per_cycle = num_atoms if mc_steps is None else int(mc_steps)

    sampler = WCSamplingHybridSampler(
        atoms=atoms,
        calculator=calculator,
        temperature=temperature,
        md_temperature=(md_temperature if md_temperature is not None else temperature),
        steps=cycles,
        md_steps_per_cycle=md_steps,
        mc_steps_per_cycle=mc_steps_per_cycle,
        wc_shells=wc_shells,
        species_pair=species_pair,
        aggregate=aggregate,
        verbose=verbose,
    )
    sampler.run_hybrid_mcmc()
    return sampler.wc_series, sampler.atoms


def save_series_csv(path: Path, series: Sequence[float]) -> None:
    """Save a numeric time series to CSV with a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write("step,metric\n")
        for idx, val in enumerate(series):
            f.write(f"{idx},{val}\n")


def plot_series_per_size(out_png: Path, size_to_series: Dict[int, Sequence[float]]) -> None:
    """Plot WC series for each size on a single figure."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for n, series in sorted(size_to_series.items()):
        plt.plot(series, label=f"{n}x{n}x{n}", linewidth=1)
    plt.xlabel("MD step index (across all cycles)")
    plt.ylabel("WC metric")
    plt.title("Warren–Cowley metric vs MD steps for supercell sizes")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint.

    Parses arguments, builds calculators and supercells, runs the hybrid
    sampler, computes WC per MD step, and saves plots/CSVs.
    """
    p = argparse.ArgumentParser(description="Hybrid MD+MC WC-parameter sweep over supercell sizes")
    p.add_argument("--model", required=True, help="Path to Allegro/NequIP/MACE model file")
    p.add_argument("--structure", default=None, help="Optional path to input structure (e.g., POSCAR, CIF). If omitted, builds bcc V primitive and randomly substitutes Cr/Ti/W/Zr at given at%.")
    p.add_argument("--size-min", type=int, required=True, help="Minimum replication factor n (supercell n x n x n)")
    p.add_argument("--size-max", type=int, required=True, help="Maximum replication factor n (inclusive)")
    p.add_argument("--backend", default="allegro", help="Calculator backend: allegro|mace|auto (default: allegro)")
    p.add_argument("--device", default="cpu", help="Device: cpu|cuda (default: cpu)")
    p.add_argument("--species-map", type=Path, default=None, help="Optional JSON with backend kwargs (e.g., species_to_type_name)")
    p.add_argument("--cycles", type=int, default=15, help="Hybrid cycles (default: 15)")
    p.add_argument("--md-steps", type=int, default=1000, help="MD steps per cycle (default: 1000)")
    p.add_argument("--mc-steps", default="natoms", help="MC steps per cycle; 'natoms' or integer (default: natoms)")
    p.add_argument("--temperature", type=float, default=1000.0, help="MC temperature K (default: 1000)")
    p.add_argument("--md-temperature", type=float, default=None, help="MD temperature K (default: same as MC)")
    p.add_argument("--num-shells", type=int, default=1, help="Number of WC shells to infer (default: 1)")
    p.add_argument("--pair", default=None, help="Optional species pair like 'Cu,Ni' to plot WC_{i,j}")
    p.add_argument("--aggregate", default="mean_abs_offdiag", help="Aggregation when pair is not set")
    p.add_argument("--outdir", type=Path, default=Path("./wc_sweep_out"), help="Output directory for plots/CSVs")
    p.add_argument("--verbose", action="store_true", help="Enable progress bars")
    # Build-structure controls
    p.add_argument("--a", type=float, default=3.01, help="Lattice parameter a for bcc V when building (default: 3.01 Å)")
    p.add_argument("--cr-pct", type=float, default=1.5, help="Target at% for Cr (floored) when building (default: 1.5)")
    p.add_argument("--ti-pct", type=float, default=2.0, help="Target at% for Ti (floored) when building (default: 2.0)")
    p.add_argument("--w-pct", type=float, default=4.0, help="Target at% for W (floored) when building (default: 4.0)")
    p.add_argument("--zr-pct", type=float, default=0.5, help="Target at% for Zr (floored) when building (default: 0.5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for substitutions (default: 42)")
    args = p.parse_args(argv)

    # Build or load base structure
    if args.structure is None:
        # Build bcc V primitive (cubic=False)
        atoms: Atoms = bulk('V', 'bcc', a=float(args.a), cubic=False)
    else:
        structure_path = Path(args.structure)
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")
        atoms = read(str(structure_path))
        if not isinstance(atoms, Atoms):
            raise ValueError("Failed to read a single ASE Atoms object from the structure path.")

    calc = build_calculator(
        model=args.model,
        device=args.device,
        backend=args.backend,
        species_map=args.species_map,
        extra_kwargs={},
    )

    species_pair: Optional[Tuple[str, str]] = None
    if args.pair:
        toks = [t.strip() for t in args.pair.split(",")]
        if len(toks) != 2:
            raise ValueError("--pair must be provided as 'A,B'")
        species_pair = (toks[0], toks[1])

    mc_steps_val: Optional[int]
    if isinstance(args.mc_steps, str) and args.mc_steps.lower() == 'natoms':
        mc_steps_val = None
    else:
        mc_steps_val = int(args.mc_steps)

    size_to_series: Dict[int, List[float]] = {}
    rng = np.random.default_rng(int(args.seed))
    for n in range(int(args.size_min), int(args.size_max) + 1):
        # Prepare supercell and apply random substitutions if building
        tmp_atoms = atoms.copy().repeat((n, n, n))
        if args.structure is None:
            _apply_random_substitutions(
                tmp_atoms,
                target_atpercent={
                    'Cr': float(args.cr_pct),
                    'Ti': float(args.ti_pct),
                    'W': float(args.w_pct),
                    'Zr': float(args.zr_pct),
                },
                host_symbol='V',
                rng=rng,
            )
        # Infer shells on the actual geometry
        wc_shells = infer_wc_shells(tmp_atoms, num_shells=int(args.num_shells))

        series, _ = run_for_size(
            base_atoms=(atoms if args.structure is not None else bulk('V', 'bcc', a=float(args.a), cubic=False)),
            n=n,
            calculator=calc,
            cycles=int(args.cycles),
            md_steps=int(args.md_steps),
            mc_steps=mc_steps_val,
            temperature=float(args.temperature),
            md_temperature=(None if args.md_temperature is None else float(args.md_temperature)),
            wc_shells=wc_shells,
            species_pair=species_pair,
            aggregate=str(args.aggregate),
            verbose=bool(args.verbose),
        )
        size_to_series[n] = series

        # Save CSV per size
        csv_path = args.outdir / f"wc_series_n{n}.csv"
        save_series_csv(csv_path, series)

    # Plot overlay of all sizes
    plot_path = args.outdir / "wc_series_overlay.png"
    plot_series_per_size(plot_path, size_to_series)


if __name__ == "__main__":
    main()


