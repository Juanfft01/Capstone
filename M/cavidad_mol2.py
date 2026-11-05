import re
from pathlib import Path

import numpy as np

from CageCavityCalc.CageCavityCalc import cavity
from CageCavityCalc.data import atom_mass, vdw_radii


def elemento(tipo, nombre):
    base = tipo.split(".")[0].upper()

    candidatos = [clave for clave in atom_mass if base.startswith(clave)]
    if candidatos:
        return max(candidatos, key=len)

    coincidencia = re.match(r"([A-Za-z]+)", nombre)
    if coincidencia:
        base = coincidencia.group(1).upper()
        candidatos = [clave for clave in atom_mass if base.startswith(clave)]
        if candidatos:
            return max(candidatos, key=len)

    return base


def leer_mol2(ruta):
    filas = []
    en_atomos = False

    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        if linea.startswith("@<TRIPOS>ATOM"):
            en_atomos = True
            continue

        if linea.startswith("@<TRIPOS>"):
            en_atomos = False
            continue

        if not en_atomos or not linea.strip():
            continue

        partes = linea.split()
        if len(partes) < 6:
            continue

        filas.append(
            {
                "nombre": partes[1],
                "posicion": (float(partes[2]), float(partes[3]), float(partes[4])),
                "tipo": partes[5],
                "resname": partes[7] if len(partes) > 7 else "",
            }
        )

    return filas


def es_agua(fila):
    nombre = fila["nombre"].upper()
    resname = fila["resname"].upper()
    return (
        resname in {"HOH", "WAT", "SOL", "H2O"}
        or nombre.startswith("OW")
        or nombre.startswith("HW")
    )


def calcular_cavidad(ruta, quitar_agua):
    filas = leer_mol2(ruta)
    if not filas:
        raise ValueError("no se encontraron atomos en el MOL2")

    if quitar_agua:
        filas = [fila for fila in filas if not es_agua(fila)]

    if not filas:
        raise ValueError("no quedaron atomos despues de quitar el agua")

    posiciones = np.array([fila["posicion"] for fila in filas], dtype=float)
    nombres = [elemento(fila["tipo"], fila["nombre"]) for fila in filas]

    cav = cavity()
    cav.positions = posiciones
    cav.atom_names = nombres
    cav.atom_masses = np.array([atom_mass.get(n, 0.0) for n in nombres])
    cav.atom_vdw = np.array([vdw_radii.get(n, 0.0) for n in nombres])
    cav.n_atoms = len(posiciones)

    desconocidos = sorted({n for n in nombres if n not in atom_mass})
    if desconocidos:
        print(f"Advertencia: elementos no reconocidos: {', '.join(desconocidos)}")

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

    etiqueta = f"{ruta.stem} (sin agua)" if quitar_agua else ruta.stem

    if volume is None:
        print(f"{etiqueta}: {cav.n_atoms} atomos; no se detecto cavidad")
    else:
        print(f"{etiqueta}: {cav.n_atoms} atomos; volumen = {volume:.3f} A^3")

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
    ruta = BASE / "2517249.mol2"
    quitar_agua = False

    if not ruta.exists():
        print(f"No existe el archivo: {ruta}")
        return 1

    try:
        cav = calcular_cavidad(ruta, quitar_agua)
        base = ruta.with_name(ruta.stem + ("_sin_agua" if quitar_agua else ""))
        guardar_salida(cav, base)
    except Exception as exc:
        print(f"Error durante el calculo: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
