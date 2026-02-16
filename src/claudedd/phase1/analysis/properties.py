"""Physicochemical property calculation.

Computes essential drug discovery properties: MW, LogP, TPSA, QED, etc.
"""

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from claudedd.core.logging import get_logger, progress_bar
from claudedd.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("analysis.properties")


@dataclass
class PhysicoChemProperties:
    """Container for key physicochemical properties."""
    molecular_weight: float = 0.0
    exact_molecular_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    ring_count: int = 0
    aromatic_ring_count: int = 0
    fraction_csp3: float = 0.0
    heavy_atom_count: int = 0
    qed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "MolWt": self.molecular_weight,
            "ExactMolWt": self.exact_molecular_weight,
            "LogP": self.logp,
            "TPSA": self.tpsa,
            "HBD": self.hbd,
            "HBA": self.hba,
            "NumRotatableBonds": self.rotatable_bonds,
            "RingCount": self.ring_count,
            "NumAromaticRings": self.aromatic_ring_count,
            "FractionCSP3": self.fraction_csp3,
            "HeavyAtomCount": self.heavy_atom_count,
            "QED": self.qed,
        }


def compute_properties(mol: Chem.Mol) -> PhysicoChemProperties:
    """Compute the standard set of physicochemical properties."""
    return PhysicoChemProperties(
        molecular_weight=Descriptors.MolWt(mol),
        exact_molecular_weight=Descriptors.ExactMolWt(mol),
        logp=Crippen.MolLogP(mol),
        tpsa=Descriptors.TPSA(mol),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
        ring_count=Descriptors.RingCount(mol),
        aromatic_ring_count=rdMolDescriptors.CalcNumAromaticRings(mol),
        fraction_csp3=Descriptors.FractionCSP3(mol),
        heavy_atom_count=Lipinski.HeavyAtomCount(mol),
        qed=QED.qed(mol),
    )


def compute_properties_dataset(
    dataset: MoleculeDataset,
) -> MoleculeDataset:
    """Compute physicochemical properties for all valid records."""
    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Computing properties"):
        if rec.mol is None:
            continue
        try:
            props = compute_properties(rec.mol)
            rec.properties.update(props.to_dict())
            rec.add_provenance("properties:computed")
            count += 1
        except Exception as e:
            logger.warning(f"Property computation failed for {rec.record_id}: {e}")

    logger.info(f"Computed properties for {count} molecules")
    return dataset
