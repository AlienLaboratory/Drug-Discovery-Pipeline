"""Docking box (grid) definition and validation.

Defines the search space for molecular docking around a ligand
or at specified coordinates.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem

from drugflow.core.constants import GRID_PADDING
from drugflow.core.exceptions import DockingError

logger = logging.getLogger(__name__)


@dataclass
class DockingBox:
    """Defines a 3D docking search box."""

    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    @property
    def center(self) -> Tuple[float, float, float]:
        return (self.center_x, self.center_y, self.center_z)

    @property
    def size(self) -> Tuple[float, float, float]:
        return (self.size_x, self.size_y, self.size_z)

    @property
    def volume(self) -> float:
        return self.size_x * self.size_y * self.size_z

    def to_dict(self) -> dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
        }


def define_grid_from_ligand(
    ligand: Chem.Mol,
    padding: float = GRID_PADDING,
    conf_id: int = 0,
) -> DockingBox:
    """Define docking box around a ligand.

    Args:
        ligand: Ligand molecule with 3D coords.
        padding: Extra space around ligand (Angstroms per side).
        conf_id: Conformer ID.

    Returns:
        DockingBox centered on the ligand.
    """
    if ligand is None or ligand.GetNumConformers() == 0:
        raise DockingError("Ligand has no 3D conformers for grid definition")

    conf = ligand.GetConformer(conf_id)
    positions = []
    for i in range(ligand.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])

    positions = np.array(positions)
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)

    center = (min_coords + max_coords) / 2.0
    size = (max_coords - min_coords) + 2 * padding

    return DockingBox(
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(size[2]),
    )


def define_grid_from_coords(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
) -> DockingBox:
    """Define docking box from explicit coordinates.

    Args:
        center: (x, y, z) center coordinates.
        size: (x, y, z) box dimensions.

    Returns:
        DockingBox.
    """
    return DockingBox(
        center_x=center[0],
        center_y=center[1],
        center_z=center[2],
        size_x=size[0],
        size_y=size[1],
        size_z=size[2],
    )


def validate_grid(box: DockingBox) -> bool:
    """Validate a docking box has reasonable dimensions.

    Args:
        box: DockingBox to validate.

    Returns:
        True if valid.
    """
    # Check sizes are positive
    if box.size_x <= 0 or box.size_y <= 0 or box.size_z <= 0:
        return False

    # Check sizes are reasonable (not too small or too large)
    for s in [box.size_x, box.size_y, box.size_z]:
        if s < 5.0 or s > 200.0:
            return False

    return True
