# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Optional, List, Dict, Tuple
import pickle
from pathlib import Path

import torch

from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.data.misc import chemical_symbols_to_atomic_numbers_dict
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.nn.utils import scatter, with_edge_vectors_
from nequip.utils.compile import conditional_torchscript_jit


def _build_nlh_tables(
    type_names: List[str],
    chemical_species: List[str],
    coeff_dict: Dict[Tuple[int, int], Tuple[float, ...]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds flattened coefficient tables for the NLH potential.

    Args:
        type_names (List[str]): List of type names.
        chemical_species (List[str]): List of chemical symbols.
        coeff_dict (Dict[Tuple[int, int], Tuple[float, ...]]): Dictionary of coefficients.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Flattened a and b coefficient tables.
    """
    num_types = len(type_names)
    atomic_numbers = [
        chemical_symbols_to_atomic_numbers_dict[s] for s in chemical_species
    ]
    z_to_type_idx = {z: i for i, z in enumerate(atomic_numbers)}

    a_tensor = torch.full(
        (num_types, num_types, 3), float("nan"), dtype=torch.get_default_dtype()
    )
    b_tensor = torch.full(
        (num_types, num_types, 3), float("nan"), dtype=torch.get_default_dtype()
    )

    for (z1, z2), coeffs in coeff_dict.items():
        if z1 in z_to_type_idx and z2 in z_to_type_idx:
            i, j = z_to_type_idx[z1], z_to_type_idx[z2]
            a_tensor[i, j] = torch.as_tensor(coeffs[:3])
            b_tensor[i, j] = torch.as_tensor(coeffs[3:])

    # Permute and flatten
    a_flat = a_tensor.permute(2, 0, 1).reshape(3, -1)
    b_flat = b_tensor.permute(2, 0, 1).reshape(3, -1)
    return a_flat, b_flat


class _NLH(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Z: torch.Tensor,
        r: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        qqr2exesquare: float,
        a_coeffs_flat: torch.Tensor,
        b_coeffs_flat: torch.Tensor,
    ) -> torch.Tensor:
        num_types = Z.shape[0]
        # Get atom types for each edge
        edge_types_unshaped = torch.index_select(
            atom_types, 0, edge_index.reshape(-1)
        )
        edge_types = edge_types_unshaped.view(2, -1)
        type_i, type_j = edge_types[0], edge_types[1]

        # Select Z for each edge
        Z_unshaped = torch.index_select(Z, 0, edge_types_unshaped)
        Zi, Zj = Z_unshaped.view(2, -1)
        
        # Look up coefficients
        idx = type_i * num_types + type_j
        a_k = a_coeffs_flat.T[idx]  # [n_edge, 3]
        b_k = b_coeffs_flat.T[idx]  # [n_edge, 3]

        # Compute phi(r)
        # r is [n_edge,], we need [n_edge, 1] to broadcast with coeffs
        r_bc = r.unsqueeze(-1)
        phi_r = torch.sum(a_k * torch.exp(-b_k * r_bc), dim=-1)

        # Compute energy
        # Add a small epsilon to r to avoid division by zero
        eng = qqr2exesquare * (Zi * Zj / (r + 1e-12)) * phi_r
        return eng


@compile_mode("script")
class NLH(GraphModuleMixin, torch.nn.Module):
    """`NLH <https://doi.org/10.1103/PhysRevA.111.032818>`_ pair potential energy term.

    Args:
        type_names (List[str]): list of type names known by the model, ``[atom1, atom2, atom3]``
        chemical_species (List[str]): list of chemical symbols, e.g. ``[C, H, O]``
        units (str): `LAMMPS units <https://docs.lammps.org/units.html>`_ that the data is in; ``metal`` and ``real`` are presently supported -- raise a GitHub issue if more is desired
    """

    def __init__(
        self,
        type_names: List[str],
        chemical_species: List[str],
        units: str,
        irreps_in=None,
    ):
        super().__init__()
        num_types = len(type_names)
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        )
        assert len(chemical_species) == num_types
        atomic_numbers_list: List[int] = [
            chemical_symbols_to_atomic_numbers_dict[chemical_species[type_i]]
            for type_i in range(num_types)
        ]
        if min(atomic_numbers_list) < 1:
            raise ValueError(
                f"Your chemical symbols don't seem valid (minimum atomic number is {min(atomic_numbers_list)} < 1); did you try to use fake chemical symbols for arbitrary atom types?"
            )
        
        # Determine the path to the pickle file relative to this file
        base_path = Path(__file__).parent
        pickle_path = base_path / "nlh_coeffs.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"NLH coefficient pickle file not found at {pickle_path}. "
                "Please run scratch/scripts/make_nlh_tables.py to generate it."
            )
        with open(pickle_path, "rb") as f:
            coeff_dict = pickle.load(f)
        
        a_coeffs_flat, b_coeffs_flat = _build_nlh_tables(
            type_names, chemical_species, coeff_dict
        )

        # LAMMPS note on units:
        # > The numerical values of the exponential decay constants in the
        # > screening function depend on the unit of distance. In the above
        # > equation they are given for units of Angstroms. LAMMPS will
        # > automatically convert these values to the distance unit of the
        # > specified LAMMPS units setting. The values of Z should always be
        # > given as multiples of a proton's charge, e.g. 29.0 for copper.
        # So, we store the atomic numbers directly.
        self.register_buffer(
            "atomic_numbers",
            torch.as_tensor(atomic_numbers_list, dtype=torch.get_default_dtype()),
        )
        self.register_buffer("a_coeffs_flat", a_coeffs_flat)
        self.register_buffer("b_coeffs_flat", b_coeffs_flat)

        # And we have to convert our value of prefector into the model's physical units
        # Here, prefactor is (electron charge)^2 / (4 * pi * electrical permisivity of vacuum)
        # we have a value for that in eV and Angstrom
        # See https://github.com/lammps/lammps/blob/c415385ab4b0983fa1c72f9e92a09a8ed7eebe4a/src/update.cpp#L187 for values from LAMMPS
        # LAMMPS uses `force->qqr2e * force->qelectron * force->qelectron`
        # Make it a buffer so rescalings are persistent, it still acts as a scalar Tensor
        self.register_buffer(
            "_qqr2exesquare",
            torch.as_tensor(
                {"metal": 14.399645 * (1.0) ** 2, "real": 332.06371 * (1.0) ** 2}[
                    units
                ],
                dtype=torch.float64,
            )
            * 0.5,  # Put half the energy on each of ij, ji
        )
        self._nlh = conditional_torchscript_jit(_NLH())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """"""
        data = with_edge_vectors_(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        nlh_edge_eng = self._nlh(
            Z=self.atomic_numbers,
            r=data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1),
            atom_types=data[AtomicDataDict.ATOM_TYPE_KEY],
            edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
            qqr2exesquare=self._qqr2exesquare,
            a_coeffs_flat=self.a_coeffs_flat,
            b_coeffs_flat=self.b_coeffs_flat,
        ).unsqueeze(-1)
        # apply cutoff
        nlh_edge_eng = nlh_edge_eng * data[AtomicDataDict.EDGE_CUTOFF_KEY]
        atomic_eng = scatter(
            nlh_edge_eng,
            edge_center,
            dim=0,
            dim_size=AtomicDataDict.num_nodes(data),
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            atomic_eng = atomic_eng + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_eng
        return data


__all__ = [NLH]