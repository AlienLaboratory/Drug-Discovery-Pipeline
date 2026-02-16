"""Protein-ligand contact detection.

Detects hydrogen bonds, hydrophobic contacts, pi-stacking,
and salt bridges based on distance and geometric criteria.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem

from drugflow.core.constants import (
    HBOND_ACCEPTORS,
    HBOND_DONORS,
    HYDROPHOBIC_ELEMENTS,
    INTERACTION_HBOND_ANGLE,
    INTERACTION_HBOND_DISTANCE,
    INTERACTION_HYDROPHOBIC_DISTANCE,
    INTERACTION_PI_STACKING_DISTANCE,
    INTERACTION_SALT_BRIDGE_DISTANCE,
)
from drugflow.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def _get_positions(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    """Get all atom positions as numpy array."""
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers")
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        positions[i] = [pos.x, pos.y, pos.z]
    return positions


def detect_hbonds(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    distance_cutoff: float = INTERACTION_HBOND_DISTANCE,
    conf_id_lig: int = 0,
    conf_id_prot: int = 0,
) -> List[Dict]:
    """Detect hydrogen bonds between ligand and protein.

    A hydrogen bond is defined as donor...acceptor distance < cutoff
    where donor = N, O atom with H, acceptor = N, O, F, S.

    Args:
        ligand: Ligand molecule with 3D coords.
        protein: Protein molecule with 3D coords.
        distance_cutoff: Max donor-acceptor distance (Angstroms).
        conf_id_lig: Ligand conformer ID.
        conf_id_prot: Protein conformer ID.

    Returns:
        List of H-bond dicts with ligand_atom, protein_atom, distance, type.
    """
    lig_pos = _get_positions(ligand, conf_id_lig)
    prot_pos = _get_positions(protein, conf_id_prot)

    hbonds = []

    for i in range(ligand.GetNumAtoms()):
        atom_i = ligand.GetAtomWithIdx(i)
        elem_i = atom_i.GetAtomicNum()
        if elem_i not in HBOND_DONORS and elem_i not in HBOND_ACCEPTORS:
            continue

        for j in range(protein.GetNumAtoms()):
            atom_j = protein.GetAtomWithIdx(j)
            elem_j = atom_j.GetAtomicNum()
            if elem_j not in HBOND_DONORS and elem_j not in HBOND_ACCEPTORS:
                continue

            dist = np.linalg.norm(lig_pos[i] - prot_pos[j])
            if dist > distance_cutoff:
                continue

            # Determine donor/acceptor roles
            is_donor_i = elem_i in HBOND_DONORS and atom_i.GetTotalNumHs() > 0
            is_acceptor_j = elem_j in HBOND_ACCEPTORS
            is_donor_j = elem_j in HBOND_DONORS and atom_j.GetTotalNumHs() > 0
            is_acceptor_i = elem_i in HBOND_ACCEPTORS

            hb_type = None
            if is_donor_i and is_acceptor_j:
                hb_type = "ligand_donor"
            elif is_donor_j and is_acceptor_i:
                hb_type = "ligand_acceptor"

            if hb_type:
                hbonds.append({
                    "ligand_atom": i,
                    "protein_atom": j,
                    "distance": float(dist),
                    "type": hb_type,
                    "ligand_element": atom_i.GetSymbol(),
                    "protein_element": atom_j.GetSymbol(),
                })

    return hbonds


def detect_hydrophobic_contacts(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    distance_cutoff: float = INTERACTION_HYDROPHOBIC_DISTANCE,
    conf_id_lig: int = 0,
    conf_id_prot: int = 0,
) -> List[Dict]:
    """Detect hydrophobic contacts between ligand and protein.

    Args:
        ligand: Ligand molecule.
        protein: Protein molecule.
        distance_cutoff: Max distance for hydrophobic contact.
        conf_id_lig: Ligand conformer ID.
        conf_id_prot: Protein conformer ID.

    Returns:
        List of contact dicts.
    """
    lig_pos = _get_positions(ligand, conf_id_lig)
    prot_pos = _get_positions(protein, conf_id_prot)

    contacts = []

    # Find hydrophobic atoms (carbon, not charged, not in polar groups)
    lig_hydrophobic = [
        i for i in range(ligand.GetNumAtoms())
        if ligand.GetAtomWithIdx(i).GetAtomicNum() in HYDROPHOBIC_ELEMENTS
        and ligand.GetAtomWithIdx(i).GetTotalNumHs() >= 0
    ]
    prot_hydrophobic = [
        j for j in range(protein.GetNumAtoms())
        if protein.GetAtomWithIdx(j).GetAtomicNum() in HYDROPHOBIC_ELEMENTS
    ]

    for i in lig_hydrophobic:
        for j in prot_hydrophobic:
            dist = np.linalg.norm(lig_pos[i] - prot_pos[j])
            if dist <= distance_cutoff:
                contacts.append({
                    "ligand_atom": i,
                    "protein_atom": j,
                    "distance": float(dist),
                    "type": "hydrophobic",
                })

    return contacts


def detect_pi_stacking(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    distance_cutoff: float = INTERACTION_PI_STACKING_DISTANCE,
    conf_id_lig: int = 0,
    conf_id_prot: int = 0,
) -> List[Dict]:
    """Detect pi-stacking interactions between aromatic rings.

    Args:
        ligand: Ligand molecule.
        protein: Protein molecule.
        distance_cutoff: Max distance between ring centroids.

    Returns:
        List of pi-stacking contact dicts.
    """
    lig_pos = _get_positions(ligand, conf_id_lig)
    prot_pos = _get_positions(protein, conf_id_prot)

    contacts = []

    # Find aromatic ring centroids
    lig_ring_info = ligand.GetRingInfo()
    prot_ring_info = protein.GetRingInfo()

    lig_centroids = []
    for ring in lig_ring_info.AtomRings():
        if ligand.GetAtomWithIdx(ring[0]).GetIsAromatic():
            centroid = np.mean(lig_pos[list(ring)], axis=0)
            lig_centroids.append((ring, centroid))

    prot_centroids = []
    for ring in prot_ring_info.AtomRings():
        if protein.GetAtomWithIdx(ring[0]).GetIsAromatic():
            centroid = np.mean(prot_pos[list(ring)], axis=0)
            prot_centroids.append((ring, centroid))

    for lig_ring, lig_cent in lig_centroids:
        for prot_ring, prot_cent in prot_centroids:
            dist = np.linalg.norm(lig_cent - prot_cent)
            if dist <= distance_cutoff:
                contacts.append({
                    "ligand_ring": list(lig_ring),
                    "protein_ring": list(prot_ring),
                    "centroid_distance": float(dist),
                    "type": "pi_stacking",
                })

    return contacts


def detect_salt_bridges(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    distance_cutoff: float = INTERACTION_SALT_BRIDGE_DISTANCE,
    conf_id_lig: int = 0,
    conf_id_prot: int = 0,
) -> List[Dict]:
    """Detect salt bridge interactions (charge-charge).

    Args:
        ligand: Ligand molecule.
        protein: Protein molecule.
        distance_cutoff: Max distance.

    Returns:
        List of salt bridge dicts.
    """
    lig_pos = _get_positions(ligand, conf_id_lig)
    prot_pos = _get_positions(protein, conf_id_prot)

    contacts = []

    # Find charged atoms
    lig_charged = [
        (i, ligand.GetAtomWithIdx(i).GetFormalCharge())
        for i in range(ligand.GetNumAtoms())
        if ligand.GetAtomWithIdx(i).GetFormalCharge() != 0
    ]
    prot_charged = [
        (j, protein.GetAtomWithIdx(j).GetFormalCharge())
        for j in range(protein.GetNumAtoms())
        if protein.GetAtomWithIdx(j).GetFormalCharge() != 0
    ]

    for i, charge_i in lig_charged:
        for j, charge_j in prot_charged:
            if charge_i * charge_j < 0:  # Opposite charges
                dist = np.linalg.norm(lig_pos[i] - prot_pos[j])
                if dist <= distance_cutoff:
                    contacts.append({
                        "ligand_atom": i,
                        "protein_atom": j,
                        "distance": float(dist),
                        "ligand_charge": charge_i,
                        "protein_charge": charge_j,
                        "type": "salt_bridge",
                    })

    return contacts


def detect_all_contacts(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    conf_id_lig: int = 0,
    conf_id_prot: int = 0,
) -> Dict[str, List[Dict]]:
    """Detect all types of protein-ligand contacts.

    Args:
        ligand: Ligand molecule.
        protein: Protein molecule.

    Returns:
        Dict mapping contact type to list of contacts.
    """
    return {
        "hbonds": detect_hbonds(ligand, protein, conf_id_lig=conf_id_lig, conf_id_prot=conf_id_prot),
        "hydrophobic": detect_hydrophobic_contacts(ligand, protein, conf_id_lig=conf_id_lig, conf_id_prot=conf_id_prot),
        "pi_stacking": detect_pi_stacking(ligand, protein, conf_id_lig=conf_id_lig, conf_id_prot=conf_id_prot),
        "salt_bridges": detect_salt_bridges(ligand, protein, conf_id_lig=conf_id_lig, conf_id_prot=conf_id_prot),
    }
