"""2D molecular structure visualization using RDKit's drawing utilities."""

from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from claudedd.core.logging import get_logger
from claudedd.core.models import MoleculeDataset

logger = get_logger("viz.structure")


def draw_molecule(
    mol: Chem.Mol,
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (400, 300),
    highlight_atoms: Optional[List[int]] = None,
    highlight_bonds: Optional[List[int]] = None,
    legend: str = "",
) -> Optional[bytes]:
    """Draw a single molecule as a 2D image."""
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)

    if output_path and output_path.lower().endswith(".svg"):
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    drawer.drawOptions().addStereoAnnotation = True

    if highlight_atoms or highlight_bonds:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms or [],
            highlightBonds=highlight_bonds or [],
            legend=legend,
        )
    else:
        drawer.DrawMolecule(mol, legend=legend)

    drawer.FinishDrawing()
    data = drawer.GetDrawingText()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if output_path.lower().endswith(".svg") else "wb"
        with open(output_path, mode) as f:
            f.write(data)
        return None

    return data if isinstance(data, bytes) else data.encode()


def draw_molecule_grid(
    dataset: MoleculeDataset,
    output_path: str,
    mols_per_page: int = 20,
    cols: int = 5,
    sub_img_size: Tuple[int, int] = (300, 200),
    legends: Optional[List[str]] = None,
    property_label: Optional[str] = None,
) -> str:
    """Draw a grid of molecules from a dataset."""
    valid = [r for r in dataset.valid_records if r.mol is not None]
    mols = []
    labels = []

    for i, rec in enumerate(valid[:mols_per_page]):
        mol = rec.mol
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        mols.append(mol)

        if legends and i < len(legends):
            labels.append(legends[i])
        elif property_label and property_label in rec.properties:
            val = rec.properties[property_label]
            labels.append(f"{property_label}={val:.2f}" if isinstance(val, float) else str(val))
        else:
            labels.append(rec.source_id or rec.smiles[:25])

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=sub_img_size,
        legends=labels,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Saved molecule grid ({len(mols)} mols) to {output_path}")
    return output_path
