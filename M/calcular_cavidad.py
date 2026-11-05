from pathlib import Path

from CageCavityCalc.CageCavityCalc import cavity

BASE = Path(__file__).resolve().parent
CAGE = BASE / "6COX.pdb"


def main():
    cav = cavity()
    cav.read_file(str(CAGE))

    window_radius = cav.calculate_window()
    cav.distance_threshold_for_90_deg_angle = max(window_radius * 2.0, 5.0)
    cav.grid_spacing = 1.0
    cav.dummy_atom_radii = 1.0

    volume = cav.calculate_volume()

    print(f"Atoms: {cav.n_atoms}")
    print(f"Window radius: {window_radius:.3f} A")
    print(f"Cavity volume: {volume:.3f} A^3")

    cav.print_to_file(str(BASE / "ncage_cavity.pdb"))
    cav.print_to_pymol(str(BASE / "ncage_cavity.pml"))


if __name__ == "__main__":
    main()
