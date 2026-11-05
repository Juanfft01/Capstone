from pathlib import Path

import MDAnalysis as mda
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from CageCavityCalc.CageCavityCalc import cavity


def preparar_mol(mol):
    if mol.GetNumConformers() > 0 and mol.GetConformer().Is3D():
        return mol

    print("  SDF sin coordenadas 3D: generando conformero con RDKit...")
    mol = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol = Chem.AddHs(mol)

    resultado = AllChem.EmbedMolecule(mol, randomSeed=42)
    if resultado != 0:
        raise ValueError("RDKit no pudo generar coordenadas 3D")

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception:
            pass

    return mol


def universe_from_mol(mol):
    conf = mol.GetConformer()
    names = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=float,
    )

    u = mda.Universe.empty(
        len(names),
        n_residues=1,
        atom_resindex=[0] * len(names),
        trajectory=True,
    )
    u.add_TopologyAttr("names", names)
    u.atoms.positions = positions
    return u


def calcular_cavidad(u, nombre):
    cav = cavity()
    cav.read_mdanalysis(u)

    window_radius = cav.calculate_window()
    cav.distance_threshold_for_90_deg_angle = max(window_radius * 2.0, 5.0)
    cav.grid_spacing = 1.0
    cav.dummy_atom_radii = 1.0

    try:
        volume = cav.calculate_volume()
    except ValueError as exc:
        if "data must be of shape" not in str(exc):
            raise
        volume = None

    if volume is None:
        print(f"{nombre}: {cav.n_atoms} atomos; no se detecto cavidad")
    else:
        print(f"{nombre}: {cav.n_atoms} atomos; volumen = {volume:.3f} A^3")

    return cav


def guardar_salida(cav, base):
    pdb = base.with_name(base.name + "_cavidad.pdb")
    pml = base.with_name(base.name + "_cavidad.pml")

    cav.print_to_file(str(pdb))

    with open(pml, "w", encoding="utf-8") as archivo:
        archivo.write(f'load "{pdb}"\n')
        archivo.write("extract cavity, resname CV\n")
        archivo.write(f"alter name D, vdw={cav.dummy_atom_radii}\n")
        archivo.write("show_as surface, cavity\n")

    print(f"  Guardado: {pdb.name}, {pml.name}")


def main():
    BASE = Path(__file__).resolve().parent
    ruta = BASE / "Structure2D_COMPOUND_CID_441336.sdf"

    if not ruta.exists():
        print(f"No existe el archivo: {ruta}")
        return 1

    supplier = Chem.SDMolSupplier(str(ruta), removeHs=False, sanitize=False)
    moleculas = [mol for mol in supplier if mol is not None]

    if not moleculas:
        print("No se encontraron moleculas validas en el SDF.")
        return 1

    unica = len(moleculas) == 1

    for i, mol in enumerate(moleculas, start=1):
        nombre = ruta.stem if unica else f"{ruta.stem}_mol{i}"
        try:
            mol = preparar_mol(mol)
            u = universe_from_mol(mol)
            cav = calcular_cavidad(u, nombre)
            guardar_salida(cav, ruta.with_name(nombre))
        except Exception as exc:
            print(f"{nombre}: error durante el calculo: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
