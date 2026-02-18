"""BCL-2 Docking Study: End-to-end molecular docking workflow.

Steps:
1. Load novel candidates from data/bcl2_senolytic/novel_candidates.csv
2. Run ADMET predictions on all 397 candidates
3. Filter to ADMET-favorable + high potency candidates
4. Download BCL-2 crystal structure (PDB: 6O0K)
5. Prepare protein using DrugFlow protein_prep
6. Define docking grid from co-crystallized ligand position
7. Prepare 3D conformers for top candidates
8. Attempt AutoDock Vina docking (or fallback to shape screening)
9. Generate integrated report

Usage:
    python scripts/bcl2_docking_study.py [--output-dir DIR] [--top-n N]
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from drugflow.core.logging import get_logger, setup_logging
from drugflow.core.models import MoleculeDataset, MoleculeRecord

setup_logging(level="INFO")
logger = get_logger("bcl2_docking_study")

# BCL-2 binding site approximate coordinates from PDB 6O0K
# These are the centroid of the Venetoclax co-crystallized ligand
BCL2_BINDING_SITE_CENTER = (14.0, 32.0, 6.0)
BCL2_BINDING_SITE_SIZE = (30.0, 30.0, 30.0)


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def download_pdb(pdb_id, output_path):
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Saved to {output_path} ({size_kb:.0f} KB)")
        return output_path
    except Exception as e:
        print(f"  WARNING: Could not download PDB {pdb_id}: {e}")
        print("  Continuing without protein structure...")
        return None


def load_candidates(candidates_path):
    """Step 1: Load novel candidates."""
    print_header("Step 1: Loading Novel Candidates")
    df = pd.read_csv(candidates_path)
    print(f"  Loaded {len(df)} candidates from {candidates_path}")

    # Identify SMILES column
    smiles_col = None
    for col in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    # Build dataset
    records = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is not None:
            rec = MoleculeRecord(mol=mol, record_id=f"cand_{idx}")
            rec.smiles = row[smiles_col]
            # Carry over existing properties
            for col in df.columns:
                if col != smiles_col:
                    rec.properties[col] = row[col]
            records.append(rec)

    dataset = MoleculeDataset(records=records)
    print(f"  Valid molecules: {len(dataset.valid_records)}")
    return dataset, df


def run_admet_predictions(dataset, output_dir):
    """Step 2: Run ADMET predictions on all candidates."""
    print_header("Step 2: Running ADMET Predictions")
    from drugflow.phase5.admet.pipeline import predict_admet_dataset

    dataset = predict_admet_dataset(dataset)

    # Summarize
    scored = [r for r in dataset.valid_records if "admet_score" in r.properties]
    scores = [r.properties["admet_score"] for r in scored]
    classes = {}
    for r in scored:
        cls = r.properties.get("admet_class", "unknown")
        classes[cls] = classes.get(cls, 0) + 1

    print(f"  ADMET predictions completed for {len(scored)} molecules")
    print(f"  Score: mean={sum(scores)/len(scores):.3f}, "
          f"min={min(scores):.3f}, max={max(scores):.3f}")
    for cls in ["favorable", "moderate", "poor"]:
        n = classes.get(cls, 0)
        pct = 100 * n / len(scored)
        print(f"    {cls}: {n} ({pct:.1f}%)")

    # Save full ADMET results
    admet_path = os.path.join(output_dir, "admet_all_candidates.csv")
    rows = []
    for rec in scored:
        row = {"smiles": rec.smiles or Chem.MolToSmiles(rec.mol)}
        row.update({k: v for k, v in rec.properties.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(admet_path, index=False)
    print(f"  Saved ADMET results to {admet_path}")

    # Save ADMET summary
    summary = {
        "total_molecules": len(scored),
        "mean_score": round(sum(scores) / len(scores), 3),
        "class_distribution": classes,
        "score_range": [round(min(scores), 3), round(max(scores), 3)],
    }
    summary_path = os.path.join(output_dir, "admet_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return dataset


def select_docking_candidates(dataset, top_n=50, min_admet_score=0.4, min_pic50=7.0):
    """Step 3: Filter and select top candidates for docking."""
    print_header("Step 3: Selecting Docking Candidates")

    candidates = []
    for rec in dataset.valid_records:
        if "admet_score" not in rec.properties:
            continue

        admet_score = rec.properties["admet_score"]
        admet_class = rec.properties.get("admet_class", "unknown")

        # Get predicted potency
        pic50 = None
        for key in ["predicted_pIC50", "pIC50", "predicted_pic50"]:
            if key in rec.properties and rec.properties[key] is not None:
                try:
                    pic50 = float(rec.properties[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Filter: not poor ADMET
        if admet_class == "poor":
            continue

        # Filter: minimum predicted potency (if available)
        if pic50 is not None and pic50 < min_pic50:
            continue

        # Combined ranking score
        if pic50 is not None:
            # Normalize pIC50 to 0-1 range (assume 5-10 range)
            pic50_norm = max(0, min(1, (pic50 - 5.0) / 5.0))
            combined = 0.4 * admet_score + 0.6 * pic50_norm
        else:
            combined = admet_score

        candidates.append((rec, combined, admet_score, pic50))

    # Sort by combined score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Select top N
    selected = candidates[:top_n]

    print(f"  Passed filters: {len(candidates)} molecules")
    print(f"  Selected top {len(selected)} for docking")

    if selected:
        print(f"\n  Top 5 candidates:")
        for rec, comb, admet, pic50 in selected[:5]:
            sid = rec.properties.get("source", rec.record_id)
            pic50_str = f"pIC50={pic50:.2f}" if pic50 else "pIC50=N/A"
            print(f"    {sid}: combined={comb:.3f}, ADMET={admet:.3f}, {pic50_str}")

    selected_records = [s[0] for s in selected]
    return MoleculeDataset(records=selected_records)


def prepare_protein(pdb_path, output_dir):
    """Step 5: Prepare protein structure."""
    print_header("Step 5: Preparing Protein Structure")

    if pdb_path is None or not os.path.exists(pdb_path):
        print("  No protein structure available — skipping protein prep")
        return None

    try:
        from drugflow.phase4.docking.protein_prep import prepare_protein as prep_protein
        protein = prep_protein(pdb_path, remove_waters=True, remove_hets=True)
        if protein is not None:
            n_atoms = protein.GetNumAtoms()
            print(f"  Prepared protein: {n_atoms} atoms (water + hetatoms removed)")
            return protein
        else:
            print("  WARNING: Protein preparation returned None")
            return None
    except Exception as e:
        print(f"  WARNING: Protein preparation failed: {e}")
        return None


def prepare_3d_conformers(dataset, output_dir, n_confs=10):
    """Step 7: Generate 3D conformers for docking candidates."""
    print_header("Step 7: Preparing 3D Conformers")

    prep_dir = os.path.join(output_dir, "prepared_3d")
    os.makedirs(prep_dir, exist_ok=True)

    prepared_count = 0
    failed_count = 0

    for rec in dataset.valid_records:
        if rec.mol is None:
            continue

        try:
            mol = Chem.AddHs(rec.mol)
            # Try embedding
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 1
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

            if len(cids) == 0:
                failed_count += 1
                continue

            # Optimize with MMFF
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                for cid in cids:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                    except Exception:
                        pass

            # Save SDF
            sdf_path = os.path.join(prep_dir, f"{rec.record_id}.sdf")
            writer = Chem.SDWriter(sdf_path)
            for cid in cids:
                writer.write(mol, confId=cid)
            writer.close()

            rec.mol = mol
            rec.properties["n_conformers"] = len(cids)
            prepared_count += 1

        except Exception as e:
            logger.warning(f"3D prep failed for {rec.record_id}: {e}")
            failed_count += 1

    print(f"  Prepared: {prepared_count} molecules ({n_confs} confs each)")
    print(f"  Failed: {failed_count}")
    print(f"  SDF files saved to: {prep_dir}")
    return dataset


def attempt_vina_docking(dataset, protein_path, output_dir):
    """Step 8a: Try AutoDock Vina docking."""
    print_header("Step 8: Molecular Docking")

    try:
        from drugflow.phase4.docking.vina_wrapper import _check_vina_available
        if not _check_vina_available():
            raise ImportError("Vina not available")

        from drugflow.phase4.docking.vina_wrapper import dock_dataset_vina
        from drugflow.phase4.docking.grid import DockingBox

        box = DockingBox(
            center_x=BCL2_BINDING_SITE_CENTER[0],
            center_y=BCL2_BINDING_SITE_CENTER[1],
            center_z=BCL2_BINDING_SITE_CENTER[2],
            size_x=BCL2_BINDING_SITE_SIZE[0],
            size_y=BCL2_BINDING_SITE_SIZE[1],
            size_z=BCL2_BINDING_SITE_SIZE[2],
        )

        print("  AutoDock Vina detected! Running docking...")
        dataset = dock_dataset_vina(dataset, protein_path, box)
        print("  Vina docking complete!")
        return dataset, "vina"

    except (ImportError, Exception) as e:
        print(f"  Vina not available ({e})")
        print("  Falling back to shape-based screening...")
        return None, None


def fallback_shape_screening(dataset, output_dir):
    """Step 8b: Shape screening as Vina fallback."""
    print("\n  Running shape-based screening against Venetoclax...")

    # Venetoclax SMILES (FDA-approved BCL-2 inhibitor)
    venetoclax_smiles = (
        "CC1(C)CCC(=C(c2ccc(Cl)cc2)c2ccc(cc2)-c2noc3cc(ccc23)"
        "OCCOCCOCCOCCN2CCOCC2)CC1"
    )
    ref_mol = Chem.MolFromSmiles(venetoclax_smiles)
    if ref_mol is None:
        # Use a simpler Venetoclax fragment
        venetoclax_smiles = "c1ccc(-c2ccc(-c3noc4ccccc34)cc2)c(Cl)c1"
        ref_mol = Chem.MolFromSmiles(venetoclax_smiles)

    if ref_mol is None:
        print("  WARNING: Could not parse Venetoclax reference — using fingerprint similarity")
        return _fingerprint_similarity_fallback(dataset)

    # Generate 3D for reference
    ref_mol = Chem.AddHs(ref_mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(ref_mol, params)

    if ref_mol.GetNumConformers() == 0:
        print("  WARNING: Could not generate 3D for reference — using fingerprint similarity")
        return _fingerprint_similarity_fallback(dataset)

    # Try MMFF optimization
    try:
        AllChem.MMFFOptimizeMolecule(ref_mol, maxIters=500)
    except Exception:
        pass

    # Shape screening
    try:
        from drugflow.phase4.shape_screening.scoring import compute_shape_tanimoto
        from drugflow.phase4.shape_screening.alignment import align_molecules_o3a as align_molecules

        scored = []
        n_aligned = 0
        for rec in dataset.valid_records:
            if rec.mol is None or rec.mol.GetNumConformers() == 0:
                rec.properties["shape_score"] = 0.0
                scored.append((rec, 0.0))
                continue

            try:
                aligned_mol = align_molecules(rec.mol, ref_mol)
                shape_score = compute_shape_tanimoto(aligned_mol, ref_mol)
                rec.properties["shape_score"] = round(shape_score, 3)
                scored.append((rec, shape_score))
                if shape_score > 0:
                    n_aligned += 1
            except Exception:
                rec.properties["shape_score"] = 0.0
                scored.append((rec, 0.0))

        # Sort by shape score
        scored.sort(key=lambda x: x[1], reverse=True)

        if n_aligned > 0:
            print(f"  Shape screening completed for {len(scored)} molecules")
            print(f"  Successfully aligned: {n_aligned}")
            print(f"  Top shape score: {scored[0][1]:.3f}")
            mean_shape = sum(s[1] for s in scored) / len(scored)
            print(f"  Mean shape score: {mean_shape:.3f}")
            return dataset, "shape"
        else:
            print(f"  Shape alignment produced no valid scores")
            print(f"  Falling back to fingerprint similarity...")
            return _fingerprint_similarity_fallback(dataset)

    except Exception as e:
        print(f"  Shape screening failed: {e}")
        print("  Falling back to fingerprint similarity...")
        return _fingerprint_similarity_fallback(dataset)


def _fingerprint_similarity_fallback(dataset):
    """Use Tanimoto fingerprint similarity against Venetoclax."""
    from rdkit.Chem import AllChem as AC
    from rdkit import DataStructs

    # Venetoclax reference fingerprint
    ven_smiles = "c1ccc(-c2ccc(-c3noc4ccccc34)cc2)c(Cl)c1"
    ref_mol = Chem.MolFromSmiles(ven_smiles)
    if ref_mol is None:
        ref_mol = Chem.MolFromSmiles("c1ccccc1")  # ultimate fallback

    ref_fp = AC.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

    for rec in dataset.valid_records:
        if rec.mol is None:
            rec.properties["similarity_score"] = 0.0
            continue
        try:
            fp = AC.GetMorganFingerprintAsBitVect(rec.mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(ref_fp, fp)
            rec.properties["similarity_score"] = round(sim, 3)
        except Exception:
            rec.properties["similarity_score"] = 0.0

    sims = [r.properties.get("similarity_score", 0) for r in dataset.valid_records]
    if sims:
        print(f"  Fingerprint similarity completed for {len(sims)} molecules")
        print(f"  Mean Tanimoto vs Venetoclax: {sum(sims)/len(sims):.3f}")
        print(f"  Max: {max(sims):.3f}")

    return dataset, "fingerprint_similarity"


def generate_report(dataset, output_dir, method_used, admet_df=None):
    """Step 9: Generate integrated report."""
    print_header("Step 9: Generating Integrated Report")

    # Build results DataFrame
    rows = []
    for rec in dataset.valid_records:
        row = {"smiles": rec.smiles or (Chem.MolToSmiles(rec.mol) if rec.mol else "")}
        row["record_id"] = rec.record_id

        # Key scores
        for key in ["predicted_pIC50", "predicted_IC50_nM", "admet_score", "admet_class",
                     "admet_n_red_flags", "admet_n_yellow_flags",
                     "MolWt", "LogP", "QED", "RingCount",
                     "admet_herg_risk", "admet_ames_risk", "admet_hepatotox_risk",
                     "admet_caco2_class", "admet_hia_class", "admet_bbb_penetrant",
                     "admet_metabolic_stability", "admet_cyp_inhibition_risk",
                     "shape_score", "similarity_score", "vina_score",
                     "n_conformers"]:
            if key in rec.properties:
                row[key] = rec.properties[key]

        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Determine scoring column for ranking
    if "vina_score" in results_df.columns and results_df["vina_score"].abs().sum() > 0:
        # Vina: lower is better (negative kcal/mol)
        rank_col = "vina_score"
        results_df = results_df.sort_values(rank_col, ascending=True)
    elif "similarity_score" in results_df.columns and results_df["similarity_score"].sum() > 0:
        rank_col = "similarity_score"
        results_df = results_df.sort_values(rank_col, ascending=False)
    elif "shape_score" in results_df.columns and results_df["shape_score"].sum() > 0:
        rank_col = "shape_score"
        results_df = results_df.sort_values(rank_col, ascending=False)
    else:
        # Fall back to ADMET score
        rank_col = "admet_score"
        results_df = results_df.sort_values(rank_col, ascending=False)

    results_df["rank"] = range(1, len(results_df) + 1)

    # Save results
    results_path = os.path.join(output_dir, "docking_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Results saved to {results_path}")

    # Save candidates selected for docking
    cand_path = os.path.join(output_dir, "docking_candidates.csv")
    results_df.to_csv(cand_path, index=False)

    # Study report JSON
    report = {
        "study": "BCL-2 Senolytic Drug Discovery — Docking Study",
        "target": "BCL-2 (PDB: 6O0K)",
        "method": method_used,
        "total_novel_candidates": len(results_df),
        "scoring_column": rank_col,
        "top_10": [],
    }

    for _, row in results_df.head(10).iterrows():
        entry = {
            "rank": int(row.get("rank", 0)),
            "smiles": row.get("smiles", ""),
            "admet_score": float(row.get("admet_score", 0)),
            "admet_class": str(row.get("admet_class", "")),
        }
        if "predicted_pIC50" in row and pd.notna(row.get("predicted_pIC50")):
            entry["predicted_pIC50"] = float(row["predicted_pIC50"])
        if rank_col in row and pd.notna(row.get(rank_col)):
            entry[rank_col] = float(row[rank_col])
        report["top_10"].append(entry)

    report_path = os.path.join(output_dir, "study_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Study report saved to {report_path}")

    # Print top results
    print(f"\n  Method: {method_used}")
    print(f"  Ranked by: {rank_col}")
    print(f"\n  Top 10 Candidates:")
    print(f"  {'Rank':>4}  {'ADMET':>6}  {'Class':>10}  ", end="")
    if "predicted_pIC50" in results_df.columns:
        print(f"{'pIC50':>6}  ", end="")
    print(f"{'Score':>8}  SMILES")
    print("  " + "-" * 70)

    for _, row in results_df.head(10).iterrows():
        rank = int(row.get("rank", 0))
        admet = row.get("admet_score", 0)
        cls = row.get("admet_class", "?")
        smi = str(row.get("smiles", ""))[:40]
        score_val = row.get(rank_col, 0)

        line = f"  {rank:>4}  {admet:>6.3f}  {cls:>10}  "
        if "predicted_pIC50" in results_df.columns and pd.notna(row.get("predicted_pIC50")):
            line += f"{row['predicted_pIC50']:>6.2f}  "
        elif "predicted_pIC50" in results_df.columns:
            line += f"{'N/A':>6}  "
        line += f"{score_val:>8.3f}  {smi}"
        print(line)

    return report


def main():
    parser = argparse.ArgumentParser(description="BCL-2 Docking Study")
    parser.add_argument("--output-dir", default="data/bcl2_senolytic/docking_results",
                        help="Output directory for results")
    parser.add_argument("--candidates", default="data/bcl2_senolytic/novel_candidates.csv",
                        help="Path to novel candidates CSV")
    parser.add_argument("--pdb-id", default="6O0K",
                        help="PDB ID for BCL-2 structure")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top candidates to dock")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PDB download")
    parser.add_argument("--skip-admet", action="store_true",
                        help="Skip ADMET predictions")
    parser.add_argument("--n-confs", type=int, default=5,
                        help="Number of 3D conformers per molecule")
    args = parser.parse_args()

    print_header("BCL-2 Senolytic Drug Discovery — Docking Study")
    print(f"  Target: BCL-2 (PDB: {args.pdb_id})")
    print(f"  Candidates: {args.candidates}")
    print(f"  Output: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load candidates
    dataset, original_df = load_candidates(args.candidates)

    # Step 2: ADMET predictions
    if not args.skip_admet:
        dataset = run_admet_predictions(dataset, args.output_dir)
    else:
        print("\n  Skipping ADMET predictions")

    # Step 3: Select docking candidates
    docking_dataset = select_docking_candidates(dataset, top_n=args.top_n)

    if len(docking_dataset.valid_records) == 0:
        print("\n  ERROR: No candidates passed filters!")
        print("  Try lowering --top-n or adjusting filter thresholds")
        return

    # Step 4: Download protein structure
    pdb_path = os.path.join(args.output_dir, f"{args.pdb_id}.pdb")
    if not args.skip_download:
        print_header("Step 4: Downloading Protein Structure")
        pdb_path = download_pdb(args.pdb_id, pdb_path)
    elif os.path.exists(pdb_path):
        print(f"\n  Using existing PDB: {pdb_path}")
    else:
        pdb_path = None

    # Step 5: Prepare protein
    protein_mol = prepare_protein(pdb_path, args.output_dir) if pdb_path else None

    # Step 6: Grid definition
    print_header("Step 6: Defining Docking Grid")
    print(f"  Using known BCL-2 binding site coordinates:")
    print(f"  Center: {BCL2_BINDING_SITE_CENTER}")
    print(f"  Size: {BCL2_BINDING_SITE_SIZE}")

    # Step 7: Prepare 3D conformers
    docking_dataset = prepare_3d_conformers(
        docking_dataset, args.output_dir, n_confs=args.n_confs
    )

    # Step 8: Docking or shape screening
    method_used = "none"
    if pdb_path and protein_mol:
        result, method = attempt_vina_docking(docking_dataset, pdb_path, args.output_dir)
        if result is not None:
            docking_dataset = result
            method_used = method
        else:
            docking_dataset, method_used = fallback_shape_screening(
                docking_dataset, args.output_dir
            )
    else:
        docking_dataset, method_used = fallback_shape_screening(
            docking_dataset, args.output_dir
        )

    # Step 9: Generate report
    report = generate_report(docking_dataset, args.output_dir, method_used)

    print_header("Study Complete!")
    print(f"  Method used: {method_used}")
    print(f"  Results directory: {args.output_dir}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"    {f} ({size:.1f} KB)")
        elif os.path.isdir(fpath):
            n_files = len(os.listdir(fpath))
            print(f"    {f}/ ({n_files} files)")


if __name__ == "__main__":
    main()
