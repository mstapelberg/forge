#!/usr/bin/env python
###########################################################################################
# Script for evaluating configurations contained in an xyz file with trained model(s)
# Authors: Ilyes Batatia, Gregor Simm (original), revised by ChatGPT
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import time
import json
import numpy as np
import ase.io
import ase.data
import torch
from collections import defaultdict
import warnings
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=FutureWarning, 
                        message="You are using `torch.load` with `weights_only=False`")

from mace import data
from mace.tools import torch_geometric, torch_tools, utils

# Parse command-line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="Path to XYZ configurations", required=True)
    parser.add_argument(
        "--model",
        help="Path(s) to model file(s); specify one or more",
        nargs="+",
        required=True,
    )
    parser.add_argument("--output", help="Output XYZ path (if one model)", required=True)
    parser.add_argument(
        "--json_output",
        help="Output JSON file path for detailed results",
        default="results.json",
    )
    parser.add_argument(
        "--device",
        help="Select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="Set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="Compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="Model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ref_prefix",
        help="Prefix for the reference energy, forces and stress keys",
        type=str,
        default="REF_",
    )
    parser.add_argument(
        "--info_prefix",
        help="Prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--show_progress",
        help="Show progress bars using tqdm",
        action="store_true",
        default=False,
    )
    # New argument for warmup duplication.
    parser.add_argument(
        "--num_warmup_steps",
        help="Number of warmup steps to perform (only used if batch_size==1). "
             "These warmup steps are duplicates of the first unique structures and are ignored in final metrics.",
        type=int,
        default=150,
    )
    # New flag to save parity plots.
    parser.add_argument(
        "--save_parity_plot",
        help="Save parity plot figure for energy/atom, forces, and stresses",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load configurations (ASE atoms objects)
    atoms_list = ase.io.read(args.configs, index=":")

    # Save the true values (if present) before predictions overwrite them.
    true_energies, true_forces, true_stresses, config_types = [], [], [], []
    for atoms in atoms_list:
        true_energies.append(atoms.info.get(args.ref_prefix + "energy"))
        forces_key = args.ref_prefix + "forces"
        force_key = args.ref_prefix + "force"  # alternative key
        if forces_key in atoms.arrays:
            true_forces.append(atoms.arrays[forces_key])
        elif force_key in atoms.arrays:
            true_forces.append(atoms.arrays[force_key])
        else:
            true_forces.append(None)
        if args.compute_stress:
            true_stresses.append(atoms.info.get(args.ref_prefix + "stress"))
        config_types.append(atoms.info.get("config_type", "default"))

    all_models_results = []
    model_iterator = args.model
    if args.show_progress and len(args.model) > 1:
        model_iterator = tqdm(model_iterator, desc="Evaluating models", leave=False)
        
    for model_path in model_iterator:
        print(f"Evaluating model: {model_path}")
        # Load model, move it to device and set to train mode (to allow gradients for forces)
        model = torch.load(model_path, map_location=args.device)
        model = model.to(device)
        model.train()  # Use train mode so gradients are computed.
        for param in model.parameters():
            param.requires_grad = True
            if param.device != device:
                param.data = param.data.to(device)
        if hasattr(model, 'atomic_energies_fn') and hasattr(model.atomic_energies_fn, 'atomic_energies'):
            if model.atomic_energies_fn.atomic_energies.device != device:
                model.atomic_energies_fn.atomic_energies = model.atomic_energies_fn.atomic_energies.to(device)

        # Create z_table and get optional head(s)
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        try:
            heads = model.heads
        except AttributeError:
            heads = None

        # Prepare dataset from configurations.
        config_iterator = atoms_list
        if args.show_progress and len(atoms_list) > 10:
            config_iterator = tqdm(config_iterator, desc="Preparing data", leave=False)
        dataset = [
            data.AtomicData.from_config(
                data.config_from_atoms(config),
                z_table=z_table,
                cutoff=float(model.r_max),
                heads=heads,
            )
            for config in config_iterator
        ]

        # If batch_size == 1, duplicate the first num_warmup_steps unique structures.
        if args.batch_size == 1 and args.num_warmup_steps > 0:
            num_warmup = min(args.num_warmup_steps, len(dataset))
            print(f"Adding warmup duplicates of the first {num_warmup} unique structures.")
            warmup_items = dataset[:num_warmup]
            dataset = warmup_items + dataset

        # Create data loader.
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        # --- Main Evaluation Loop ---
        energies_list = []
        forces_collection = []
        stresses_list = []
        contributions_list = []
        inference_times = []  # one per structure

        batch_iterator = data_loader
        if args.show_progress:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", leave=False)
        for batch in batch_iterator:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            batch_dict = batch.to_dict()
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(device)
            output = model(batch_dict, compute_stress=args.compute_stress)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            num_struct = int(batch.ptr.shape[0]) - 1
            per_struct_time = batch_time / num_struct if num_struct > 0 else batch_time
            inference_times.extend([per_struct_time] * num_struct)
            energies_list.append(torch_tools.to_numpy(output["energy"]))
            if args.compute_stress:
                stresses_list.append(torch_tools.to_numpy(output["stress"]))
            if args.return_contributions:
                contributions_list.append(torch_tools.to_numpy(output["contributions"]))
            forces = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            forces_collection.append(forces[:-1])  # drop empty tail

        # Consolidate predictions from all batches.
        predicted_energies = np.concatenate(energies_list, axis=0)
        predicted_forces = [f for batch_forces in forces_collection for f in batch_forces]
        if args.compute_stress:
            predicted_stresses = np.concatenate(stresses_list, axis=0)
        if args.return_contributions:
            predicted_contributions = np.concatenate(contributions_list, axis=0)

        # If warmup duplicates were added, ignore the first num_warmup_steps results.
        if args.batch_size == 1 and args.num_warmup_steps > 0:
            print(f"Ignoring the first {args.num_warmup_steps} warmup predictions.")
            predicted_energies = predicted_energies[args.num_warmup_steps:]
            predicted_forces = predicted_forces[args.num_warmup_steps:]
            inference_times = inference_times[args.num_warmup_steps:]
            if args.compute_stress:
                predicted_stresses = predicted_stresses[args.num_warmup_steps:]

        # Consistency checks.
        if not (len(atoms_list) == len(predicted_energies) == len(predicted_forces)):
            print("Warning: Mismatch in inference times and number of structures.")
        if args.compute_stress and len(atoms_list) != predicted_stresses.shape[0]:
            print("Warning: Mismatch in stress predictions and number of structures.")

        # Compute per-structure error metrics.
        per_structure_results = []
        energy_errors, force_errors, stress_errors = [], [], []
        rel_force_errors, rel_stress_errors = [], []
        for i in range(len(atoms_list)):
            n_atoms = len(atoms_list[i])
            struct_result = {
                "structure_index": i,
                "config_type": config_types[i],
                "n_atoms": n_atoms,
                "inference_time": inference_times[i],
                "predicted_energy": float(predicted_energies[i]),
                "predicted_energy_per_atom": float(predicted_energies[i]) / n_atoms,
            }
            true_E = true_energies[i]
            if true_E is not None:
                e_err = abs(float(predicted_energies[i]) - float(true_E))
                e_err_per_atom = e_err / n_atoms
                struct_result["true_energy"] = float(true_E)
                struct_result["true_energy_per_atom"] = float(true_E) / n_atoms
                struct_result["energy_error"] = e_err
                struct_result["energy_error_per_atom"] = e_err_per_atom
                energy_errors.append(e_err_per_atom)
                if abs(true_E) > 1e-8:
                    struct_result["relative_energy_error_percent"] = e_err / abs(true_E) * 100
                else:
                    struct_result["relative_energy_error_percent"] = None
            else:
                struct_result["energy_error"] = None
                struct_result["energy_error_per_atom"] = None
                struct_result["relative_energy_error_percent"] = None

            true_F = true_forces[i]
            pred_F = predicted_forces[i]
            if true_F is not None:
                try:
                    true_F = np.array(true_F)
                    pred_F = np.array(pred_F)
                    f_err = float(np.sqrt(np.mean((pred_F - true_F) ** 2)))
                    struct_result["force_error"] = f_err
                    force_errors.append(f_err)
                    f_rel_err = compute_rel_rmse(pred_F - true_F, true_F)
                    struct_result["relative_force_error_percent"] = f_rel_err
                    rel_force_errors.append(f_rel_err)
                except Exception as e:
                    print(f"Warning: Could not compute force error for structure {i}: {e}")
                    struct_result["force_error"] = None
                    struct_result["relative_force_error_percent"] = None
            else:
                struct_result["force_error"] = None
                struct_result["relative_force_error_percent"] = None

            if args.compute_stress:
                true_S = true_stresses[i] if i < len(true_stresses) else None
                if true_S is not None:
                    try:
                        true_S = np.array(true_S)
                        pred_S = predicted_stresses[i].flatten()
                        s_err = float(np.sqrt(np.mean((pred_S - true_S) ** 2)))
                        struct_result["stress_error"] = s_err
                        stress_errors.append(s_err)
                        s_rel_err = compute_rel_rmse(pred_S - true_S, true_S)
                        struct_result["relative_stress_error_percent"] = s_rel_err
                        rel_stress_errors.append(s_rel_err)
                        struct_result["true_stress"] = true_S.tolist()
                    except Exception as e:
                        print(f"Warning: Could not compute stress error for structure {i}: {e}")
                        struct_result["stress_error"] = None
                        struct_result["relative_stress_error_percent"] = None
                else:
                    struct_result["stress_error"] = None
                    struct_result["relative_stress_error_percent"] = None

            per_structure_results.append(struct_result)

        # Aggregate metrics by config type.
        config_agg = defaultdict(lambda: {
            "energy_errors_per_atom": [], 
            "energy_rel_errors_percent": [],
            "force_errors": [], 
            "stress_errors": [], 
            "rel_force_errors": [],
            "rel_stress_errors": [],
            "true_energies": [],
            "true_energies_per_atom": []
        })
        for res in per_structure_results:
            key = res["config_type"]
            if res["energy_error"] is not None:
                config_agg[key]["energy_errors_per_atom"].append(res["energy_error_per_atom"])
                config_agg[key]["true_energies"].append(res["true_energy"])
                config_agg[key]["true_energies_per_atom"].append(res["true_energy_per_atom"])
                if res.get("relative_energy_error_percent") is not None:
                    config_agg[key]["energy_rel_errors_percent"].append(res["relative_energy_error_percent"])
            if res["force_error"] is not None:
                config_agg[key]["force_errors"].append(res["force_error"])
                if res.get("relative_force_error_percent") is not None:
                    config_agg[key]["rel_force_errors"].append(res["relative_force_error_percent"])
            if args.compute_stress and res.get("stress_error") is not None:
                config_agg[key]["stress_errors"].append(res["stress_error"])
                if res.get("relative_stress_error_percent") is not None:
                    config_agg[key]["rel_stress_errors"].append(res["relative_stress_error_percent"])
        aggregated_metrics = {}
        for config_type, values in config_agg.items():
            agg = {}
            if values["energy_errors_per_atom"]:
                rmse_energy = np.sqrt(np.mean(np.square(values["energy_errors_per_atom"])))
                agg["RMSE_energy_per_atom_eV"] = float(rmse_energy)
                if values["energy_rel_errors_percent"]:
                    agg["average_relative_energy_error_percent"] = float(np.mean(values["energy_rel_errors_percent"]))
                else:
                    rel_errors = []
                    for err, true_val in zip(values["energy_errors_per_atom"], values["true_energies_per_atom"]):
                        if abs(true_val) > 1e-8:
                            rel_errors.append(err / abs(true_val) * 100)
                    agg["average_relative_energy_error_percent"] = float(np.mean(rel_errors)) if rel_errors else None
            if values["force_errors"]:
                rmse_force = np.sqrt(np.mean(np.square(values["force_errors"])))
                agg["RMSE_force_eV_per_Angstrom"] = float(rmse_force)
                if values["rel_force_errors"]:
                    agg["relative_force_RMSE_percent"] = float(np.mean(values["rel_force_errors"]))
            if args.compute_stress and values["stress_errors"]:
                rmse_stress = np.sqrt(np.mean(np.square(values["stress_errors"])))
                agg["RMSE_stress_eV_per_Angstrom3"] = float(rmse_stress)
                if values["rel_stress_errors"]:
                    agg["relative_stress_RMSE_percent"] = float(np.mean(values["rel_stress_errors"]))
            aggregated_metrics[config_type] = agg

        overall_avg_inference_time = float(np.mean(inference_times))
        overall_energy_rmse = float(np.sqrt(np.mean(np.square(energy_errors)))) if energy_errors else None
        overall_force_rmse = float(np.sqrt(np.mean(np.square(force_errors)))) if force_errors else None
        overall_stress_rmse = float(np.sqrt(np.mean(np.square(stress_errors)))) if (args.compute_stress and stress_errors) else None
        overall_rel_force_rmse = float(np.mean(rel_force_errors)) if rel_force_errors else None
        overall_rel_stress_rmse = float(np.mean(rel_stress_errors)) if rel_stress_errors else None
        if energy_errors and true_energies:
            rel_energy_errors = []
            for i, (err, pred_e, true_e) in enumerate(zip(energy_errors, predicted_energies, true_energies)):
                if true_e is not None and abs(true_e) > 1e-8:
                    rel_energy_errors.append((abs(pred_e - true_e) / abs(true_e)) * 100)
            overall_rel_energy_rmse = float(np.mean(rel_energy_errors)) if rel_energy_errors else None
        else:
            overall_rel_energy_rmse = None

        model_result = {
            "model_name": model_path,
            "num_structures": len(atoms_list),
            "average_inference_time_sec": overall_avg_inference_time,
            "overall_energy_RMSE_eV_per_atom": overall_energy_rmse,
            "overall_force_RMSE_eV_per_Angstrom": overall_force_rmse,
            "overall_force_relative_RMSE_percent": overall_rel_force_rmse,
            "overall_stress_RMSE_eV_per_Angstrom3": overall_stress_rmse,
            "overall_stress_relative_RMSE_percent": overall_rel_stress_rmse,
            "per_structure": per_structure_results,
            "aggregated_by_config": aggregated_metrics,
        }
        all_models_results.append(model_result)

        summary = f"Model: {model_path}\n" \
                  f"  Average Inference Time: {overall_avg_inference_time:.4f} sec\n" \
                  f"  Overall Energy RMSE: {overall_energy_rmse:.4f} eV/atom"
        if overall_rel_energy_rmse is not None:
            summary += f" ({overall_rel_energy_rmse:.2f}%)"
        summary += f"\n  Overall Force RMSE: {overall_force_rmse:.4f} eV/Å"
        if overall_rel_force_rmse is not None:
            summary += f" ({overall_rel_force_rmse:.2f}%)"
        if args.compute_stress:
            summary += f"\n  Overall Stress RMSE: {overall_stress_rmse:.4f} eV/Å³"
            if overall_rel_stress_rmse is not None:
                summary += f" ({overall_rel_stress_rmse:.2f}%)"
        print(summary)
        print("-" * 50)

        if len(args.model) == 1:
            for i, atoms in enumerate(atoms_list):
                atoms.calc = None
                n_atoms = len(atoms)
                atoms.info[args.info_prefix + "energy"] = float(predicted_energies[i])
                atoms.info[args.info_prefix + "energy_per_atom"] = float(predicted_energies[i]) / n_atoms
                atoms.arrays[args.info_prefix + "forces"] = predicted_forces[i]
                if args.compute_stress:
                    atoms.info[args.info_prefix + "stress"] = predicted_stresses[i]
                if per_structure_results[i].get("energy_error") is not None:
                    atoms.info[args.info_prefix + "energy_error"] = per_structure_results[i]["energy_error"]
                    atoms.info[args.info_prefix + "energy_error_per_atom"] = per_structure_results[i]["energy_error_per_atom"]
                if per_structure_results[i].get("force_error") is not None:
                    atoms.info[args.info_prefix + "force_error"] = per_structure_results[i]["force_error"]
            ase.io.write(args.output, images=atoms_list, format="extxyz")
            print(f"Updated configurations with predictions written to: {args.output}")

        # --- Parity Plot Generation ---
        if args.save_parity_plot:
            # Energy parity: true vs. predicted energy per atom.
            energy_true = np.array([result["true_energy_per_atom"] for result in per_structure_results])
            energy_pred = np.array([result["predicted_energy_per_atom"] for result in per_structure_results])
            # Force parity: compute RMS force per structure.
            force_true, force_pred = [], []
            for i in range(len(atoms_list)):
                if true_forces[i] is not None:
                    ftrue = np.sqrt(np.mean(np.square(np.array(true_forces[i]))))
                    fpred = np.sqrt(np.mean(np.square(np.array(predicted_forces[i]))))
                    force_true.append(ftrue)
                    force_pred.append(fpred)
            force_true = np.array(force_true)
            force_pred = np.array(force_pred)
            # Stress parity: compute RMS stress per structure (if computed).
            if args.compute_stress:
                stress_true, stress_pred = [], []
                for i in range(len(atoms_list)):
                    if true_stresses[i] is not None:
                        strue = np.sqrt(np.mean(np.square(np.array(true_stresses[i]))))
                        spred = np.sqrt(np.mean(np.square(np.array(predicted_stresses[i]))))
                        stress_true.append(strue)
                        stress_pred.append(spred)
                stress_true = np.array(stress_true)
                stress_pred = np.array(stress_pred)
            
            # Compute RMSE for each parity plot.
            rmse_energy = np.sqrt(np.mean((energy_pred - energy_true) ** 2))
            rmse_force = np.sqrt(np.mean((force_pred - force_true) ** 2))
            if args.compute_stress:
                rmse_stress = np.sqrt(np.mean((stress_pred - stress_true) ** 2))
            
            # Create 1x3 subplot for parity plots.
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            # Energy parity plot.
            axs[0].scatter(energy_true, energy_pred, alpha=0.7)
            axs[0].plot([energy_true.min(), energy_true.max()], [energy_true.min(), energy_true.max()], 'r--')
            axs[0].set_xlabel("True Energy per Atom (eV)")
            axs[0].set_ylabel("Predicted Energy per Atom (eV)")
            axs[0].set_title(f"Energy Parity Plot\nRMSE: {rmse_energy:.4f} eV/atom")
            # Force parity plot.
            axs[1].scatter(force_true, force_pred, alpha=0.7)
            axs[1].plot([force_true.min(), force_true.max()], [force_true.min(), force_true.max()], 'r--')
            axs[1].set_xlabel("True Force Norm (eV/Å)")
            axs[1].set_ylabel("Predicted Force Norm (eV/Å)")
            axs[1].set_title(f"Force Parity Plot\nRMSE: {rmse_force:.4f} eV/Å")
            # Stress parity plot.
            if args.compute_stress:
                axs[2].scatter(stress_true, stress_pred, alpha=0.7)
                axs[2].plot([stress_true.min(), stress_true.max()], [stress_true.min(), stress_true.max()], 'r--')
                axs[2].set_xlabel("True Stress Norm (eV/Å³)")
                axs[2].set_ylabel("Predicted Stress Norm (eV/Å³)")
                axs[2].set_title(f"Stress Parity Plot\nRMSE: {rmse_stress:.4f} eV/Å³")
            else:
                axs[2].axis('off')
            plt.tight_layout()
            plot_filename = f"parity_plot_{os.path.splitext(os.path.basename(model_path))[0]}.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Parity plot saved to {plot_filename}")

    # Save detailed results to JSON.
    with open(args.json_output, "w") as f:
        json.dump(all_models_results, f, sort_keys=True, indent=4, ensure_ascii=False)
    print(f"Detailed results saved to {args.json_output}")


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100


if __name__ == "__main__":
    main()