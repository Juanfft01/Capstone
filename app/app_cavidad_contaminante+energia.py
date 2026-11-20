

import html
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from CageCavityCalc.CageCavityCalc import cavity
from CageCavityCalc.data import atom_mass, vdw_radii

PT = Chem.GetPeriodicTable()
SALIDA = Path(tempfile.mkdtemp(prefix="cavidad_mof_"))
AGUA_RES = {"HOH", "WAT", "SOL", "H2O", "DOD"}
METALES = {
    "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO",
    "NI", "CU", "ZN", "GA", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD",
    "AG", "CD", "IN", "SN", "CS", "BA", "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB",
    "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU",
    "HG", "TL", "PB", "BI", "U",
}


# ----------------------------------------------------------------------------
# Elementos y radios
# ----------------------------------------------------------------------------
def es_elemento(simbolo):
    try:
        return PT.GetAtomicNumber(simbolo.capitalize()) > 0
    except Exception:
        return False


def normalizar_elemento(texto):
    """'Zn1', 'C.ar', 'CL' -> 'ZN', 'C', 'CL'."""
    letras = re.match(r"([A-Za-z]+)", texto or "")
    if not letras:
        return "X"
    letras = letras.group(1)
    if len(letras) >= 2 and es_elemento(letras[:2]):
        return letras[:2].upper()
    if es_elemento(letras[0]):
        return letras[0].upper()
    return letras.upper()


def registrar_elementos(elementos):
    """CageCavityCalc no trae todos los metales (Co, Mn, Zr, Cr...).
    Se agregan al vuelo con los valores de RDKit para que no falle ni use radio 0."""
    for e in set(elementos):
        if e in vdw_radii and e in atom_mass:
            continue
        try:
            z = PT.GetAtomicNumber(e.capitalize())
            vdw_radii.setdefault(e, float(PT.GetRvdw(z)))
            atom_mass.setdefault(e, float(PT.GetAtomicWeight(z)))
        except Exception:
            vdw_radii.setdefault(e, 1.7)
            atom_mass.setdefault(e, 12.0)


def radios(elementos):
    registrar_elementos(elementos)
    return np.array([vdw_radii[e] for e in elementos], dtype=float)


# ----------------------------------------------------------------------------
# Lectura de archivos -> (posiciones Nx3, elementos, resnames)
# ----------------------------------------------------------------------------
def leer_mol2(ruta):
    pos, elem, res = [], [], []
    en_atomos = False
    for linea in Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines():
        if linea.startswith("@<TRIPOS>"):
            en_atomos = linea.startswith("@<TRIPOS>ATOM")
            continue
        p = linea.split()
        if not en_atomos or len(p) < 6:
            continue
        pos.append([float(p[2]), float(p[3]), float(p[4])])
        tipo = p[5].split(".")[0]
        elem.append(normalizar_elemento(tipo) if es_elemento(tipo) else normalizar_elemento(p[1]))
        res.append(p[7].upper() if len(p) > 7 else "")
    return np.array(pos), elem, res


def leer_pdb(ruta):
    pos, elem, res = [], [], []
    for linea in Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines():
        if linea.startswith(("ENDMDL", "END")) and pos:
            break
        if not linea.startswith(("ATOM", "HETATM")):
            continue
        pos.append([float(linea[30:38]), float(linea[38:46]), float(linea[46:54])])
        col_elem = linea[76:78].strip() if len(linea) >= 78 else ""
        if col_elem and es_elemento(col_elem):
            elem.append(col_elem.upper())
        else:
            nombre = linea[12:16]
            # convencion PDB: elementos de 2 letras empiezan en la columna 13
            if nombre[0] != " " and es_elemento(nombre[:2].strip()):
                elem.append(nombre[:2].strip().upper())
            else:
                elem.append(normalizar_elemento(nombre.strip()))
        res.append(linea[17:20].strip().upper())
    return np.array(pos), elem, res


def leer_xyz(ruta):
    lineas = Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(lineas[0].split()[0])
    pos, elem = [], []
    for linea in lineas[2:2 + n]:
        p = linea.split()
        elem.append(normalizar_elemento(p[0]))
        pos.append([float(x) for x in p[1:4]])
    return np.array(pos), elem, [""] * len(elem)


def preparar_mol(mol):
    """Igual que cavidad_sdf.py: si no hay 3D se genera un conformero con RDKit."""
    if mol.GetNumConformers() > 0 and mol.GetConformer().Is3D():
        return mol
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) != 0:
            raise ValueError("RDKit no pudo generar coordenadas 3D")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception:
            pass
    return mol


def mol_a_arrays(mol):
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    elem = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    return pos, elem, [""] * len(elem)


def leer_sdf(ruta):
    sup = Chem.SDMolSupplier(str(ruta), removeHs=False, sanitize=False)
    mols = [m for m in sup if m is not None]
    if not mols:
        mol = Chem.MolFromMolFile(str(ruta), removeHs=False, sanitize=False)
        mols = [mol] if mol is not None else []
    if not mols:
        raise ValueError("no se encontraron moleculas validas en el archivo")
    return mol_a_arrays(preparar_mol(mols[0]))


def leer_cif(ruta):
    import gemmi

    try:
        ss = gemmi.read_small_structure(str(ruta))
        sitios = ss.get_all_unit_cell_sites()
        if sitios:
            pos, elem = [], []
            for s in sitios:
                c = ss.cell.orthogonalize(s.fract)
                pos.append([c.x, c.y, c.z])
                elem.append(s.element.name.upper())
            return np.array(pos), elem, [""] * len(elem)
    except Exception:
        pass
    st = gemmi.read_structure(str(ruta))
    pos, elem, res = [], [], []
    for cadena in st[0]:
        for r in cadena:
            for a in r:
                pos.append([a.pos.x, a.pos.y, a.pos.z])
                elem.append(a.element.name.upper())
                res.append(r.name.upper())
    return np.array(pos), elem, res


def leer_estructura(ruta):
    ext = Path(ruta).suffix.lower()
    if ext == ".mol2":
        datos = leer_mol2(ruta)
    elif ext in (".pdb", ".ent"):
        datos = leer_pdb(ruta)
    elif ext in (".sdf", ".mol"):
        datos = leer_sdf(ruta)
    elif ext == ".xyz":
        datos = leer_xyz(ruta)
    elif ext in (".cif", ".mmcif"):
        datos = leer_cif(ruta)
    else:
        raise ValueError(f"formato no soportado: {ext}")
    if len(datos[0]) == 0:
        raise ValueError("no se encontraron atomos en el archivo")
    return datos


def quitar_agua(pos, elem, res):
    """Quita moleculas de agua: por nombre de residuo, o atomos O cuyos unicos
    vecinos (< 1.25 A) son H y que no estan unidos a C/N/S/P."""
    pos = np.asarray(pos)
    quitar = {i for i, r in enumerate(res) if r in AGUA_RES}
    arbol = cKDTree(pos)
    for i, e in enumerate(elem):
        if e != "O" or i in quitar:
            continue
        vecinos = [j for j in arbol.query_ball_point(pos[i], 1.75) if j != i]
        pesados = [j for j in vecinos if elem[j] not in ("H", "D") and elem[j] not in METALES]
        if not pesados:
            quitar.add(i)
            quitar.update(j for j in vecinos if elem[j] in ("H", "D")
                          and np.linalg.norm(pos[j] - pos[i]) < 1.25)
    keep = [i for i in range(len(elem)) if i not in quitar]
    return pos[keep], [elem[i] for i in keep], [res[i] for i in keep], len(quitar)


# ----------------------------------------------------------------------------
# Energias: lectura automatica desde el archivo y energia de adsorcion
# ----------------------------------------------------------------------------
# factores de conversion a kcal/mol
A_KCAL = {
    "Hartree": 627.509474,
    "eV": 23.060548,
    "Ry": 313.754737,
    "kcal/mol": 1.0,
    "kJ/mol": 1.0 / 4.184,
}
UNIDADES_E = list(A_KCAL)
UNIDAD_DEFECTO = "Hartree"

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"
# "Energy: -12.3", "energy = -12.3", "Energy (kcal/mol): -12.3", "E_tot -12.3"
_RE_ENERGIA = re.compile(
    r"(?i)(?:\benerg(?:y|ia|ía)\b|\bE_?(?:tot|total|scf|dft)\b)"
    r"[ \t]*(?:\([^)\n]*\))?[ \t]*[=:]?[ \t]*(" + _NUM + ")")
# solo para la linea de comentario de un .xyz (ORCA: "E -76.2", CP2K: "E = -17.1")
_RE_E_SOLA = re.compile(r"(?<![\w.])E(?:[ \t]*[=:][ \t]*|[ \t]+)(" + _NUM + ")")
_RE_NUM_SOLO = re.compile(r"\s*(" + _NUM + r")(?:\s+(\S+))?\s*")
_RE_CIF = re.compile(r"(?im)^[ \t]*(_\S*energ\S*)\s+(" + _NUM + ")")
_PATRONES_UNIDAD = [
    (r"kcal", "kcal/mol"),
    (r"(?<![a-z])kj", "kJ/mol"),
    (r"(?<![a-z])ev(?![a-z])", "eV"),
    (r"hartree|(?<![a-z])(?:eh|ha|au)(?![a-z])|a\.u\.", "Hartree"),
    (r"rydberg|(?<![a-z])ry(?![a-z])", "Ry"),
]


def _unidad_en(texto):
    t = (texto or "").lower()
    for patron, unidad in _PATRONES_UNIDAD:
        if re.search(patron, t):
            return unidad
    return None


def _a_float(txt):
    return float(txt.replace("D", "E").replace("d", "e"))


def _buscar_energia(texto, e_sola=False):
    for rx in [_RE_ENERGIA] + ([_RE_E_SOLA] if e_sola else []):
        m = rx.search(texto)
        if m:
            return _a_float(m.group(1)), _unidad_en(texto[m.start():m.end() + 12])
    return None


def _energia_xyz(texto):
    lineas = texto.splitlines()
    if len(lineas) < 2:
        return None
    comentario = lineas[1]
    m = _RE_NUM_SOLO.fullmatch(comentario)  # la linea es solo el numero (y quiza la unidad)
    if m:
        return _a_float(m.group(1)), _unidad_en(m.group(2))
    return _buscar_energia(comentario, e_sola=True)


def _energia_pdb(texto):
    cab = "\n".join(l for l in texto.splitlines()
                    if l.startswith(("REMARK", "TITLE", "HEADER", "COMPND", "COMMENT", "EXPDTA")))
    return _buscar_energia(cab)


def _energia_sdf(texto):
    lineas = texto.splitlines()
    props = []
    for i, l in enumerate(lineas):
        if l.startswith("$$$$"):
            break
        m = re.match(r">\s*(?:\d+\s*)?<([^>]+)>", l)
        if m and i + 1 < len(lineas):
            props.append((m.group(1), lineas[i + 1].strip()))
    for nombre, valor in props:
        if re.search(r"(?i)energ|^e(_?(tot|total|scf|dft))?$", nombre.strip()):
            n = _RE_NUM_SOLO.match(valor)
            if n:
                return _a_float(n.group(1)), _unidad_en(nombre + " " + valor)
    return _buscar_energia(lineas[0]) if lineas else None


def _energia_cif(texto):
    m = _RE_CIF.search(texto)
    if m:
        return _a_float(m.group(2)), _unidad_en(m.group(1) + " " + texto[m.end():m.end() + 12])
    return _buscar_energia("\n".join(l for l in texto.splitlines() if l.lstrip().startswith("#")))


def leer_energia(ruta):
    """Busca la energia dentro del archivo. Devuelve (valor, unidad o None) o None."""
    ext = Path(ruta).suffix.lower()
    try:
        texto = Path(ruta).read_text(encoding="utf-8", errors="replace")
        if ext == ".xyz":
            return _energia_xyz(texto)
        if ext in (".pdb", ".ent"):
            return _energia_pdb(texto)
        if ext in (".sdf", ".mol"):
            return _energia_sdf(texto)
        if ext in (".cif", ".mmcif"):
            return _energia_cif(texto)
        return _buscar_energia(texto)  # .mol2 y otros
    except Exception:
        return None


def auto_energia(archivo, etiqueta):
    """Callback de gr.File: devuelve (valor, unidad, mensaje) para los campos de energia."""
    if archivo is None:
        return None, gr.update(), ""
    ruta = archivo if isinstance(archivo, str) else archivo.name
    res = leer_energia(ruta)
    if res is None:
        return None, gr.update(), (f"No se encontró energía en el archivo {etiqueta}; "
                                   "escríbela manualmente.")
    valor, unidad = res
    if unidad:
        return valor, unidad, f"Energía {etiqueta} leída del archivo: {valor} {unidad}"
    return valor, gr.update(), (f"Energía {etiqueta} leída del archivo: {valor} "
                                "(el archivo no indica la unidad: revisa el selector)")


def calcular_energia_adsorcion(e_mof, u_mof, e_mol, u_mol, e_cpx, u_cpx):
    """E_ads = E(complejo) - E(MOF) - E(contaminante)."""
    if e_mof is None or e_mol is None or e_cpx is None:
        raise gr.Error("Completa las energías del MOF, del contaminante y del complejo "
                       "(MOF + contaminante).")
    mof = float(e_mof) * A_KCAL[u_mof]
    mol = float(e_mol) * A_KCAL[u_mol]
    cpx = float(e_cpx) * A_KCAL[u_cpx]
    ads = cpx - mof - mol

    def fila(nombre, kcal):
        return [nombre, f"{kcal:.4f}", f"{kcal * 4.184:.4f}", f"{kcal / A_KCAL['eV']:.4f}"]

    tabla = [fila("E MOF", mof), fila("E contaminante", mol), fila("E complejo", cpx),
             fila("E adsorción", ads)]
    if ads < 0:
        veredicto = "✅ **Adsorción favorable** (energía de adsorción negativa)."
    else:
        veredicto = "❌ **Adsorción no favorable** (energía de adsorción positiva)."
    md = (f"## {veredicto}\n\n**E_ads = {ads:.3f} kcal/mol = {ads * 4.184:.3f} kJ/mol = "
          f"{ads / A_KCAL['eV']:.4f} eV**\n\n"
          "_E_ads = E(complejo) − E(MOF) − E(contaminante). No incluye correcciones de "
          "punto cero ni BSSE; las tres energías deben venir del mismo método y nivel de teoría._")
    return md, tabla


# ----------------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------------
def ejes_principales(pos):
    centro = pos.mean(axis=0)
    if len(pos) < 3:
        return centro, np.eye(3)
    _, _, vt = np.linalg.svd(pos - centro, full_matrices=False)
    ejes = vt
    if np.linalg.det(ejes) < 0:
        ejes[2] *= -1
    return centro, ejes


def dimensiones(pos, r, ejes):
    """Extension (A) a lo largo de cada eje incluyendo radios de vdW, ordenada."""
    proy = pos @ ejes.T
    ext = (proy + r[:, None]).max(axis=0) - (proy - r[:, None]).min(axis=0)
    return np.sort(ext)


def volumen_vdw(pos, r, paso=0.25):
    """Volumen de vdW por rejilla (union de esferas)."""
    lo = (pos - r[:, None]).min(axis=0)
    hi = (pos + r[:, None]).max(axis=0)
    ejes = [np.arange(lo[k], hi[k] + paso, paso) for k in range(3)]
    g = np.stack(np.meshgrid(*ejes, indexing="ij"), axis=-1).reshape(-1, 3)
    dentro = np.zeros(len(g), dtype=bool)
    for p, ri in zip(pos, r):
        dentro |= np.sum((g - p) ** 2, axis=1) < ri * ri
    return dentro.sum() * paso ** 3


# ----------------------------------------------------------------------------
# Paso 1: cavidad del MOF
# ----------------------------------------------------------------------------
def calcular_cavidad_mof(archivo, sin_agua, grid_spacing):
    if archivo is None:
        raise gr.Error("Sube primero el archivo del MOF.")
    ruta = archivo if isinstance(archivo, str) else archivo.name
    pos, elem, res = leer_estructura(ruta)
    n_agua = 0
    if sin_agua:
        pos, elem, res, n_agua = quitar_agua(pos, elem, res)
    r_mof = radios(elem)

    cav = cavity()
    cav.positions = np.asarray(pos, dtype=float)
    cav.atom_names = list(elem)
    cav.atom_masses = np.array([atom_mass[e] for e in elem])
    cav.atom_vdw = r_mof
    cav.n_atoms = len(pos)
    cav.filename = ruta

    try:
        window_radius = float(cav.calculate_window())
    except Exception:
        window_radius = float("nan")
    umbral = window_radius * 2.0 if np.isfinite(window_radius) else 5.0
    if not np.isfinite(window_radius) or window_radius <= 0:
        window_radius = 0.0  # ventana cerrada / sin abertura detectable
    cav.distance_threshold_for_90_deg_angle = max(umbral, 5.0)
    cav.grid_spacing = float(grid_spacing)
    cav.dummy_atom_radii = float(grid_spacing)

    try:
        volumen = cav.calculate_volume()
    except ValueError as exc:
        if "data must be of shape" not in str(exc):
            raise
        volumen = None

    dummies = np.array(cav.dummy_atoms_positions, dtype=float).reshape(-1, 3)
    if volumen is None or len(dummies) == 0:
        raise gr.Error(f"No se detecto cavidad en {Path(ruta).name} ({cav.n_atoms} atomos).")

    # dimensiones de la cavidad y esfera inscrita mas grande
    centro, ejes = ejes_principales(dummies)
    dims = dimensiones(dummies, np.full(len(dummies), cav.dummy_atom_radii), ejes)
    arbol = cKDTree(pos)
    d, idx = arbol.query(dummies, k=min(8, len(pos)))
    d, idx = np.atleast_2d(d), np.atleast_2d(idx)
    libre = (d - r_mof[idx]).min(axis=1)
    i_max = int(np.argmax(libre))

    nombre = Path(ruta).stem
    pdb_cav = SALIDA / f"{nombre}_cavidad.pdb"
    pml_cav = SALIDA / f"{nombre}_cavidad.pml"
    cav.print_to_file(str(pdb_cav))
    with open(pml_cav, "w", encoding="utf-8") as f:
        f.write(f'load "{pdb_cav.name}"\nextract cavity, resname CV\n'
                f"alter name D, vdw={cav.dummy_atom_radii}\nshow_as surface, cavity\n")

    estado = {
        "nombre": nombre,
        "pos": np.asarray(pos, dtype=float),
        "elem": list(elem),
        "r": r_mof,
        "dummies": dummies,
        "dummy_r": float(cav.dummy_atom_radii),
        "volumen": float(volumen),
        "centro": centro,
        "dims": dims,
        "di": float(2 * libre[i_max]),
        "centro_di": dummies[i_max],
        "window_r": window_radius,
    }

    tabla = [
        ["Atomos del MOF", f"{len(pos)}" + (f"  ({n_agua} atomos de agua eliminados)" if sin_agua else "")],
        ["Volumen de la cavidad", f"{volumen:.1f} A^3"],
        ["Puntos de la cavidad (dummies)", f"{len(dummies)}"],
        ["Dimensiones de la cavidad (ejes principales)", " x ".join(f"{x:.1f}" for x in dims[::-1]) + " A"],
        ["Diametro de la mayor esfera inscrita (Di)", f"{estado['di']:.2f} A"],
        ["Radio de ventana (CageCavityCalc)",
         f"{window_radius:.2f} A  (diametro {2 * window_radius:.2f} A)" if window_radius > 0
         else "sin ventana abierta detectable (jaula cerrada)"],
    ]
    md = (f"### Cavidad de **{nombre}**\nVolumen = **{volumen:.1f} A^3**. "
          "Ahora sube el contaminante en el paso 2.")
    visor = visor_3d(estado)
    return estado, md, tabla, visor, [str(pdb_cav), str(pml_cav)]


# ----------------------------------------------------------------------------
# Paso 2: contaminante y prueba de ajuste
# ----------------------------------------------------------------------------
def cargar_contaminante(archivo, smiles):
    if smiles and smiles.strip():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise gr.Error("SMILES no valido.")
        pos, elem, _ = mol_a_arrays(preparar_mol(mol))
        return pos, elem, "contaminante"
    if archivo is None:
        raise gr.Error("Sube el archivo del contaminante o escribe su SMILES.")
    ruta = archivo if isinstance(archivo, str) else archivo.name
    pos, elem, _ = leer_estructura(ruta)
    return pos, elem, Path(ruta).stem


def evaluar_poses(huesped, r_h, centros, rot, arbol, r_mof, arbol_cav, tol_dentro):
    """Devuelve (max_solapamiento, fraccion_dentro) para cada combinacion centro x rotacion."""
    n = len(huesped)
    mats = rot.as_matrix().reshape(-1, 3, 3)
    rotadas = np.einsum("rij,nj->rni", mats, huesped)
    nc, nr = len(centros), len(mats)
    coords = (rotadas[None, :, :, :] + centros[:, None, None, :]).reshape(-1, 3)
    k = min(6, len(r_mof))
    d, idx = arbol.query(coords, k=k)
    d, idx = d.reshape(-1, k), idx.reshape(-1, k)
    r_rep = np.tile(r_h, nc * nr)
    solape = (r_rep[:, None] + r_mof[idx]) - d
    solape = solape.max(axis=1).reshape(nc, nr, n).max(axis=2)
    dc, _ = arbol_cav.query(coords, k=1)
    dentro = (dc.reshape(nc, nr, n) <= tol_dentro).mean(axis=2)
    return solape, dentro, rotadas


def buscar_pose(est, pos_h, r_h, n_rot=300, n_centros=40, semilla=0):
    rng = np.random.default_rng(semilla)
    arbol = cKDTree(est["pos"])
    arbol_cav = cKDTree(est["dummies"])
    tol_dentro = est["dummy_r"] * 1.5 + 0.5
    huesped = pos_h - pos_h.mean(axis=0)

    # centros candidatos: centroide de la cavidad, centro de la esfera inscrita
    # y los puntos de la cavidad mas alejados de las paredes
    d, idx = arbol.query(est["dummies"], k=min(8, len(est["pos"])))
    libre = (np.atleast_2d(d) - est["r"][np.atleast_2d(idx)]).min(axis=1)
    orden = np.argsort(-libre)[: max(1, n_centros - 2)]
    centros = np.vstack([est["centro"], est["centro_di"], est["dummies"][orden]])

    rot = Rotation.random(n_rot, random_state=semilla)
    solape, dentro, _ = evaluar_poses(huesped, r_h, centros, rot, arbol, est["r"], arbol_cav, tol_dentro)
    puntaje = np.maximum(solape, 0) + 2.0 * (1 - dentro)
    ic, ir = np.unravel_index(np.argmin(puntaje), puntaje.shape)
    mejor_c, mejor_r = centros[ic], rot[ir]
    mejor_s, mejor_d = solape[ic, ir], dentro[ic, ir]

    # refinamiento local (Monte Carlo)
    paso_t, paso_a = 0.5, 0.25
    for _ in range(300):
        c = mejor_c + rng.normal(0, paso_t, 3)
        if arbol_cav.query(c)[0] > tol_dentro:  # el centro debe quedar dentro de la cavidad
            continue
        r = Rotation.from_rotvec(rng.normal(0, paso_a, 3)) * mejor_r
        s, dn, _ = evaluar_poses(huesped, r_h, c[None], r, arbol,
                                 est["r"], arbol_cav, tol_dentro)
        s, dn = s[0, 0], dn[0, 0]
        if max(s, 0) + 2.0 * (1 - dn) < max(mejor_s, 0) + 2.0 * (1 - mejor_d):
            mejor_c, mejor_r, mejor_s, mejor_d = c, r, s, dn
    coords = mejor_r.apply(huesped) + mejor_c
    return coords, float(max(mejor_s, 0.0)), float(mejor_d)


def probar_contaminante(est, archivo, smiles, tolerancia, n_rot):
    if not est:
        raise gr.Error("Primero calcula la cavidad del MOF (paso 1).")
    pos_h, elem_h, nombre_h = cargar_contaminante(archivo, smiles)
    r_h = radios(elem_h)

    v_mol = volumen_vdw(pos_h, r_h)
    _, ejes = ejes_principales(pos_h)
    dims_h = dimensiones(pos_h, r_h, ejes)
    pc = v_mol / est["volumen"]

    coords, solape, dentro = buscar_pose(est, pos_h, r_h, n_rot=int(n_rot))
    sin_choques = solape <= tolerancia
    dims_ok = bool(np.all(dims_h <= est["dims"] + 1e-6))
    win_d = 2 * est["window_r"]
    min2 = dims_h[1]
    entra_ventana = bool(win_d > 0 and min2 <= win_d)

    if pc > 1.0:
        veredicto = "❌ **NO CABE**: el volumen del contaminante es mayor que el de la cavidad."
    elif sin_choques and dentro >= 0.95 and pc <= 0.70:
        veredicto = "✅ **SÍ CABE** dentro de la cavidad sin choques estéricos."
    elif sin_choques and dentro >= 0.95:
        veredicto = ("⚠️ **CABE, PERO AJUSTADO**: hay una pose sin choques, pero el coeficiente de "
                     "empaquetamiento es alto (> 0.70), la cavidad queda muy llena.")
    elif sin_choques:
        veredicto = (f"⚠️ **CABE PARCIALMENTE**: sin choques, pero solo el {dentro:.0%} de los átomos "
                     "queda dentro de la cavidad (parte de la molécula sobresale por una ventana).")
    else:
        veredicto = (f"❌ **NO CABE**: la mejor orientación encontrada aún choca con el armazón "
                     f"(solapamiento máximo {solape:.2f} Å > tolerancia {tolerancia:.2f} Å).")

    if win_d > 0:
        txt_ventana = (f"El diámetro crítico de la molécula (≈MIN-2, {min2:.2f} Å) es "
                       f"{'menor' if entra_ventana else 'mayor'} que el diámetro de ventana "
                       f"({win_d:.2f} Å): {'puede' if entra_ventana else 'en principio NO puede'} "
                       "entrar a la cavidad por difusión a través de la ventana.")
    else:
        txt_ventana = ("No se detectó una ventana abierta: el contaminante solo podría quedar dentro "
                       "si se encapsula durante la síntesis, no por difusión.")

    md = (f"## {veredicto}\n\n{txt_ventana}\n\n"
          "_Criterios: volumen (coeficiente de empaquetamiento, regla del 55 % de Rebek: "
          "≈0.55 óptimo, ≤0.70 aceptable), dimensiones por ejes principales y búsqueda de una "
          "orientación sin solapamiento de radios de van der Waals mayor que la tolerancia._")

    tabla = [
        ["Contaminante", f"{nombre_h} ({len(elem_h)} átomos)", ""],
        ["Volumen vdW contaminante / cavidad", f"{v_mol:.1f} / {est['volumen']:.1f} Å³", f"PC = {pc:.2f}"],
        ["Dimensiones contaminante (Å)", " x ".join(f"{x:.1f}" for x in dims_h[::-1]), ""],
        ["Dimensiones cavidad (Å)", " x ".join(f"{x:.1f}" for x in est["dims"][::-1]),
         "OK" if dims_ok else "alguna dimensión excede"],
        ["Diámetro crítico (MIN-2) vs ventana", f"{min2:.2f} vs {win_d:.2f} Å" if win_d > 0 else f"{min2:.2f} Å vs sin ventana",
         "entra" if entra_ventana else "no entra"],
        ["Máx. solapamiento vdW en la mejor pose", f"{solape:.2f} Å",
         "sin choques" if sin_choques else "choque"],
        ["Átomos dentro de la cavidad", f"{dentro:.0%}", ""],
    ]

    # archivo del complejo MOF + contaminante en la mejor pose
    pdb = SALIDA / f"{est['nombre']}_{nombre_h}_complejo.pdb"
    escribir_complejo(pdb, est, coords, elem_h)
    resumen = SALIDA / f"{est['nombre']}_{nombre_h}_resultado.json"
    resumen.write_text(json.dumps({
        "mof": est["nombre"], "contaminante": nombre_h,
        "volumen_cavidad_A3": est["volumen"], "volumen_contaminante_A3": v_mol,
        "coef_empaquetamiento": pc, "dims_cavidad_A": est["dims"].tolist(),
        "dims_contaminante_A": dims_h.tolist(), "diametro_ventana_A": win_d,
        "min2_A": float(min2), "max_solapamiento_A": solape, "fraccion_dentro": dentro,
        "veredicto": re.sub(r"[*✅❌⚠️]", "", veredicto).strip(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return md, tabla, visor_3d(est, coords, elem_h), [str(pdb), str(resumen)]


def escribir_complejo(ruta, est, coords_h, elem_h):
    lineas, n = [], 1
    for p, e in zip(est["pos"], est["elem"]):
        lineas.append(f"HETATM{n:5d} {e:<4s} MOF A   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {e:>2s}")
        n += 1
    for p, e in zip(coords_h, elem_h):
        lineas.append(f"HETATM{n:5d} {e:<4s} LIG B   2    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {e:>2s}")
        n += 1
    for p in est["dummies"]:
        lineas.append(f"HETATM{n % 100000:5d} D    CV  C   3    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          He")
        n += 1
    Path(ruta).write_text("\n".join(lineas) + "\nEND\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Visor 3D (3Dmol.js dentro de un iframe; necesita internet para cargar la libreria)
# ----------------------------------------------------------------------------
def xyz_texto(pos, elem):
    return f"{len(elem)}\n\n" + "\n".join(
        f"{e.capitalize()} {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p, e in zip(pos, elem))


def visor_3d(est, coords_h=None, elem_h=None):
    mof = json.dumps(xyz_texto(est["pos"], est["elem"]))
    cav = json.dumps(xyz_texto(est["dummies"], ["He"] * len(est["dummies"])))
    lig = json.dumps(xyz_texto(coords_h, elem_h)) if coords_h is not None else "null"
    pagina = f"""<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
<style>html,body{{margin:0;height:100%;background:#fff;font-family:sans-serif}}
#v{{width:100%;height:100%;position:relative}}
#l{{position:absolute;bottom:6px;left:8px;font-size:12px;color:#444;z-index:10}}</style></head>
<body><div id="v"></div><div id="l">gris: MOF &nbsp; azul translúcido: cavidad
{"&nbsp; esferas de color: contaminante" if coords_h is not None else ""}</div><script>
const v = $3Dmol.createViewer("v", {{backgroundColor: "white"}});
let m = v.addModel({mof}, "xyz");
m.setStyle({{}}, {{stick: {{radius: 0.12, colorscheme: "grayCarbon"}}, sphere: {{scale: 0.18}}}});
let c = v.addModel({cav}, "xyz");
c.setStyle({{}}, {{sphere: {{radius: {est['dummy_r']}, color: "#3b82f6", opacity: 0.25}}}});
const lig = {lig};
if (lig) {{ let g = v.addModel(lig, "xyz");
  g.setStyle({{}}, {{sphere: {{scale: 0.9}}, stick: {{radius: 0.2}}}}); }}
v.zoomTo(); v.render();
</script></body></html>"""
    return (f'<iframe style="width:100%;height:520px;border:1px solid #ddd;border-radius:8px" '
            f'srcdoc="{html.escape(pagina, quote=True)}"></iframe>')


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
FORMATOS_MOF = [".pdb", ".ent", ".mol2", ".sdf", ".mol", ".xyz", ".cif"]
FORMATOS_MOL = [".sdf", ".mol", ".mol2", ".pdb", ".xyz"]

with gr.Blocks(title="Cavidad de MOF y contaminantes emergentes") as demo:
    gr.Markdown("# Cavidad de MOF ↔ contaminante emergente\n"
                "1) Sube el MOF y calcula su cavidad. 2) Sube el contaminante y comprueba si cabe.")
    estado = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. MOF / jaula")
            f_mof = gr.File(label="Estructura del MOF", file_types=FORMATOS_MOF, type="filepath")
            cb_agua = gr.Checkbox(label="Quitar moléculas de agua", value=False)
            sl_grid = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Espaciado de rejilla (Å)")
            b_cav = gr.Button("Calcular cavidad", variant="primary")

            gr.Markdown("### 2. Contaminante emergente")
            f_mol = gr.File(label="Molécula del contaminante", file_types=FORMATOS_MOL, type="filepath")
            tx_smiles = gr.Textbox(label="…o SMILES (opcional)",
                                   placeholder="ej. ibuprofeno: CC(C)Cc1ccc(cc1)C(C)C(=O)O")
            sl_tol = gr.Slider(0.0, 1.0, value=0.4, step=0.05,
                               label="Tolerancia de solapamiento vdW (Å)")
            sl_rot = gr.Slider(50, 1000, value=300, step=50, label="Orientaciones a probar")
            b_fit = gr.Button("¿Cabe el contaminante?", variant="primary")

            gr.Markdown("### 3. Energía de adsorción\n"
                        "E_ads = E(complejo) − E(MOF) − E(contaminante). Si el archivo trae la "
                        "energía, el campo se llena solo; si no, escríbela.")
            with gr.Row():
                n_e_mof = gr.Number(label="Energía del MOF", value=None, precision=None, scale=3)
                u_e_mof = gr.Dropdown(UNIDADES_E, value=UNIDAD_DEFECTO, label="Unidad", scale=2)
            i_e_mof = gr.Markdown()
            with gr.Row():
                n_e_mol = gr.Number(label="Energía del contaminante", value=None, precision=None, scale=3)
                u_e_mol = gr.Dropdown(UNIDADES_E, value=UNIDAD_DEFECTO, label="Unidad", scale=2)
            i_e_mol = gr.Markdown()
            f_cpx = gr.File(label="Complejo MOF + contaminante (opcional, solo para leer su energía)",
                            file_types=FORMATOS_MOF, type="filepath")
            with gr.Row():
                n_e_cpx = gr.Number(label="Energía del complejo", value=None, precision=None, scale=3)
                u_e_cpx = gr.Dropdown(UNIDADES_E, value=UNIDAD_DEFECTO, label="Unidad", scale=2)
            i_e_cpx = gr.Markdown()
            b_ads = gr.Button("Calcular energía de adsorción", variant="primary")

        with gr.Column(scale=2):
            md_res = gr.Markdown()
            tb_cav = gr.Dataframe(headers=["Propiedad", "Valor"], label="Cavidad del MOF",
                                  interactive=False)
            tb_fit = gr.Dataframe(headers=["Criterio", "Valor", "Resultado"],
                                  label="Contaminante vs cavidad", interactive=False)
            md_ads = gr.Markdown()
            tb_ads = gr.Dataframe(headers=["Magnitud", "kcal/mol", "kJ/mol", "eV"],
                                  label="Energía de adsorción", interactive=False)
            visor = gr.HTML()
            descargas = gr.File(label="Archivos generados", file_count="multiple")

    b_cav.click(calcular_cavidad_mof, [f_mof, cb_agua, sl_grid],
                [estado, md_res, tb_cav, visor, descargas])
    b_fit.click(probar_contaminante, [estado, f_mol, tx_smiles, sl_tol, sl_rot],
                [md_res, tb_fit, visor, descargas])

    # la energia se rellena sola al subir el archivo, si el archivo la trae
    f_mof.change(lambda f: auto_energia(f, "del MOF"), f_mof, [n_e_mof, u_e_mof, i_e_mof])
    f_mol.change(lambda f: auto_energia(f, "del contaminante"), f_mol, [n_e_mol, u_e_mol, i_e_mol])
    f_cpx.change(lambda f: auto_energia(f, "del complejo"), f_cpx, [n_e_cpx, u_e_cpx, i_e_cpx])
    b_ads.click(calcular_energia_adsorcion,
                [n_e_mof, u_e_mof, n_e_mol, u_e_mol, n_e_cpx, u_e_cpx],
                [md_ads, tb_ads])


if __name__ == "__main__":
    demo.launch()
