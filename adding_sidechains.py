# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select, Superimposer
from Bio.PDB.Polypeptide import is_aa
from pdbfixer import PDBFixer
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff, Modeller
from openmm import unit, LocalEnergyMinimizer, VerletIntegrator, CustomExternalForce, Platform

# ---------------- Config ----------------
PEPTIDE_CHAIN_ID = "P"    # peptide chain id in your input PDBs
MHC_CHAIN_ID = "M"        # MHC chain id in your input PDBs
BACKBONE = {"N", "CA", "C", "O"}

# Minimization settings
K_BACKBONE = 2000.0       # kJ/mol/nm^2 restraint on peptide backbone
K_MHC = 10000.0           # kJ/mol/nm^2 restraint on ALL heavy atoms of MHC (keeps M ~fixed)
MAX_MIN_STEPS = 800       # iterations; 300-500 is faster, 800-1000 declashes more
# ---------------------------------------


def is_hydrogen_name(name: str) -> bool:
    return name.startswith("H")


class OnlyCASelect(Select):
    """Keep full MHC and only CA for peptide chain. Drops altlocs not in ' ' or 'A'."""
    def __init__(self, peptide_chain_id):
        self.chain_id = peptide_chain_id

    def accept_atom(self, atom):
        if atom.is_disordered() and atom.get_altloc() not in (" ", "A"):
            return False
        chain_id = atom.get_parent().get_parent().id
        if chain_id != self.chain_id:
            return True  # keep everything for non-peptide (e.g., MHC)
        return atom.get_name() == "CA"  # peptide: keep only CA


def extract_ca_peptide(input_pdb, output_pdb, peptide_chain_id=PEPTIDE_CHAIN_ID):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, OnlyCASelect(peptide_chain_id))


def reconstruct_with_pdbfixer(ca_only_pdb, output_pdb):
    fixer = PDBFixer(filename=ca_only_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    missing_count = sum(len(v) for v in fixer.missingAtoms.values())
    print("[DEBUG] {} - Missing atoms detected by PDBFixer: {}".format(
        os.path.basename(ca_only_pdb), missing_count))
    if missing_count > 0:
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens()
    else:
        print("[WARN] No missing atoms detected; peptide may still be CA-only if formatting confused PDBFixer.")
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def residue_key(residue):
    # residue id is like (' ', resseq, icode)
    het, resseq, icode = residue.get_id()
    return (resseq, (icode or "").strip())


def collect_atom_dict(structure, chain_id, mode: str):
    """
    mode in {'backbone', 'sidechain', 'allheavy'}
    returns dict {(reskey, atom_name) -> Atom}
    """
    d = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                rk = residue_key(res)
                for atom in res:
                    if atom.is_disordered() and atom.get_altloc() not in (" ", "A"):
                        continue
                    name = atom.get_name()
                    if is_hydrogen_name(name):
                        continue  # exclude hydrogens everywhere
                    if mode == "backbone":
                        if name in BACKBONE:
                            d[(rk, name)] = atom
                    elif mode == "sidechain":
                        if name not in BACKBONE:
                            d[(rk, name)] = atom
                    elif mode == "allheavy":
                        d[(rk, name)] = atom
    return d


def match_atom_lists(orig_struct, rebuilt_struct, orig_chain_id, rebuilt_chain_id, mode):
    d1 = collect_atom_dict(orig_struct, orig_chain_id, mode)
    d2 = collect_atom_dict(rebuilt_struct, rebuilt_chain_id, mode)
    keys = sorted(set(d1.keys()).intersection(d2.keys()))
    return [d1[k] for k in keys], [d2[k] for k in keys], len(keys)


def coords_from_atoms(atoms):
    return np.array([a.get_coord() for a in atoms], dtype=float)


def rmsd_after_rotran(fixed_xyz, moving_xyz, R, t):
    moved = moving_xyz @ R.T + t
    diff = fixed_xyz - moved
    return float(np.sqrt((diff * diff).sum() / len(fixed_xyz)))


def chain_length_map(structure):
    d = {}
    for model in structure:
        for chain in model:
            d[chain.id] = sum(1 for r in chain if is_aa(r, standard=True))
    return d


def count_std_residues(structure, chain_id):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return sum(1 for r in chain if is_aa(r, standard=True))
    return 0


def guess_rebuilt_peptide_chain_id(rebuilt_struct, target_len):
    lengths = chain_length_map(rebuilt_struct)
    best = None
    best_cid = None
    for cid, L in lengths.items():
        score = (abs(L - target_len), L)
        if best is None or score < best:
            best = score
            best_cid = cid
    print("[INFO] Rebuilt chain lengths: {} | target peptide len={} -> picked '{}'".format(
        lengths, target_len, best_cid))
    return best_cid


def compute_rmsds(original, rebuilt, orig_chain_id, rebuilt_chain_id):
    # Align on backbone heavy atoms
    bb_fix, bb_mov, n_bb = match_atom_lists(original, rebuilt, orig_chain_id, rebuilt_chain_id, "backbone")
    if n_bb < 3:
        print("[WARN] Backbone matches too few ({}) atoms; cannot align.".format(n_bb))
        return None, None, None
    sup = Superimposer()
    sup.set_atoms(bb_fix, bb_mov)
    R, t = sup.rotran
    bb_rmsd = float(sup.rms)

    sc_fix, sc_mov, n_sc = match_atom_lists(original, rebuilt, orig_chain_id, rebuilt_chain_id, "sidechain")
    sc_rmsd = rmsd_after_rotran(coords_from_atoms(sc_fix), coords_from_atoms(sc_mov), R, t) if n_sc >= 1 else None

    all_fix, all_mov, n_all = match_atom_lists(original, rebuilt, orig_chain_id, rebuilt_chain_id, "allheavy")
    all_rmsd = rmsd_after_rotran(coords_from_atoms(all_fix), coords_from_atoms(all_mov), R, t) if n_all >= 3 else None

    print("[INFO] Match counts - BB:{} SC:{} ALL:{}".format(n_bb, n_sc, n_all))
    return bb_rmsd, sc_rmsd, all_rmsd


def replace_peptide_chain(original_struct, rebuilt_struct, orig_chain_id=PEPTIDE_CHAIN_ID, rebuilt_chain_id=None):
    """Replace peptide chain in original with rebuilt one; rename back to orig_chain_id."""
    model = original_struct[0]
    if orig_chain_id in [c.id for c in model]:
        model.detach_child(orig_chain_id)
    rchain = None
    for ch in rebuilt_struct[0]:
        if ch.id == rebuilt_chain_id:
            rchain = ch
            break
    if rchain is None:
        raise ValueError("Rebuilt structure lacks chain '{}'".format(rebuilt_chain_id))
    rcopy = copy.deepcopy(rchain)
    rcopy.id = orig_chain_id  # rename back to 'P'
    model.add(rcopy)
    return original_struct


def choose_platform():
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            continue
    return Platform.getPlatform(0)


def fix_full_complex_with_pdbfixer(in_pdb, out_pdb):
    """
    Add missing heavy atoms and hydrogens to the entire merged complex.
    We DO NOT add whole missing residues (avoid hallucinated loops).
    Ensures proper terminal groups (e.g., OXT) and protonation.
    """
    fixer = PDBFixer(filename=in_pdb)
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # do not build whole residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()
    with open(out_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def minimize_with_restraints(in_pdb, out_pdb,
                             pep_chain="P", mhc_chain="M",
                             k_pep_backbone=K_BACKBONE, k_mhc=K_MHC,
                             max_iter=MAX_MIN_STEPS):
    """
    Minimize full complex with:
      - strong restraints on ALL heavy atoms of MHC (keeps M ~fixed)
      - moderate restraints on peptide BACKBONE (N, CA, C, O)
    Adds hydrogens with Modeller (handles HIS variants, OXT, etc).
    RESTRAINT PARAMETERS ARE PLAIN FLOATS: nm for positions, kJ/mol/nm^2 for k.
    """
    pdb = PDBFile(in_pdb)

    # Force field
    try:
        ff = ForceField('amber14/protein.ff14SB.xml')
    except Exception:
        ff = ForceField('amber14-all.xml')

    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=7.0)

    system = ff.createSystem(modeller.topology, constraints=None, nonbondedMethod=NoCutoff)

    # Build float parameters
    k_pep = float(k_pep_backbone)  # kJ/mol/nm^2
    k_mhc_val = float(k_mhc)       # kJ/mol/nm^2

    # Positions as plain floats in nm
    pos_nm = [p.value_in_unit(unit.nanometer) for p in modeller.positions]

    # Position restraints: energy = 0.5*k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
    restr = CustomExternalForce('0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    restr.addPerParticleParameter('k')   # kJ/mol/nm^2
    restr.addPerParticleParameter('x0')  # nm
    restr.addPerParticleParameter('y0')  # nm
    restr.addPerParticleParameter('z0')  # nm

    def is_H(name: str) -> bool:
        return name.startswith('H')

    n_mhc, n_pep = 0, 0
    for atom in modeller.topology.atoms():
        idx = atom.index
        x0, y0, z0 = pos_nm[idx]
        name = atom.name
        cid = atom.residue.chain.id
        if cid == mhc_chain and not is_H(name):
            restr.addParticle(idx, [k_mhc_val, x0, y0, z0])
            n_mhc += 1
        elif cid == pep_chain and name in BACKBONE:
            restr.addParticle(idx, [k_pep, x0, y0, z0])
            n_pep += 1
    system.addForce(restr)
    print("[INFO] Restraints added: MHC heavy={}, peptide backbone={}".format(n_mhc, n_pep))

    integrator = VerletIntegrator(0.002*unit.picoseconds)
    try:
        platform = Platform.getPlatformByName('CUDA')
    except Exception:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except Exception:
            platform = Platform.getPlatformByName('CPU')

    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Use default tolerance; avoid passing a Quantity to minimize
    LocalEnergyMinimizer.minimize(sim.context, maxIterations=max_iter)

    state = sim.context.getState(getPositions=True)
    with open(out_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)


def process_pdb(pdb_path, output_folder, chain_pep=PEPTIDE_CHAIN_ID, chain_mhc=MHC_CHAIN_ID):
    basename = os.path.basename(pdb_path)
    stripped_pdb = os.path.join(output_folder, basename.replace(".pdb", "_ca_only.pdb"))
    rebuilt_pdb  = os.path.join(output_folder, basename.replace(".pdb", "_reconstructed.pdb"))
    rebuilt_full = os.path.join(output_folder, basename.replace(".pdb", "_full_complex.pdb"))
    fixed_full   = os.path.join(output_folder, basename.replace(".pdb", "_full_complex_fixed.pdb"))
    minimized_full = os.path.join(output_folder, basename.replace(".pdb", "_full_complex_min.pdb"))

    # 1) Build CA-only peptide (keep MHC)
    extract_ca_peptide(pdb_path, stripped_pdb, chain_pep)

    # 2) Reconstruct peptide with PDBFixer
    reconstruct_with_pdbfixer(stripped_pdb, rebuilt_pdb)

    # 3) Load structures
    parser = PDBParser(QUIET=True)
    original = parser.get_structure("orig", pdb_path)
    rebuilt  = parser.get_structure("rebuild", rebuilt_pdb)

    # 4) Identify peptide chain in rebuilt (PDBFixer may rename it)
    target_len = count_std_residues(original, chain_pep)
    rebuilt_pep_id = guess_rebuilt_peptide_chain_id(rebuilt, target_len)

    # 5) Compute RMSDs (backbone-aligned; heavy atoms only)
    bb_rmsd, sc_rmsd, all_rmsd = compute_rmsds(original, rebuilt, chain_pep, rebuilt_pep_id)

    # 6) Merge rebuilt peptide back into original complex
    merged = replace_peptide_chain(original, rebuilt, chain_pep, rebuilt_pep_id)
    io = PDBIO()
    io.set_structure(merged)
    io.save(rebuilt_full)

    # 7) Fix the full complex (add missing terminal atoms and hydrogens)
    fix_full_complex_with_pdbfixer(rebuilt_full, fixed_full)

    # 8) Minimize with restraints (M fixed, peptide backbone kept)
    print("[INFO] Running restrained minimization...")
    minimize_with_restraints(fixed_full, minimized_full,
                             pep_chain=chain_pep, mhc_chain=chain_mhc,
                             k_pep_backbone=K_BACKBONE, k_mhc=K_MHC,
                             max_iter=MAX_MIN_STEPS)

    return bb_rmsd, sc_rmsd, all_rmsd


def main(input_folder):
    output_folder = input_folder.rstrip("/\\") + "_new"
    os.makedirs(output_folder, exist_ok=True)

    report_rows = []
    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(".pdb"):
            continue
        pdb_path = os.path.join(input_folder, fname)
        try:
            bb, sc, all_atoms = process_pdb(pdb_path, output_folder)
            bbs = "{:.3f}".format(bb) if bb is not None else "NA"
            scs = "{:.3f}".format(sc) if sc is not None else "NA"
            alls = "{:.3f}".format(all_atoms) if all_atoms is not None else "NA"
            print("[OK] {} | BB {} | SC {} | ALL {}".format(fname, bbs, scs, alls))
            report_rows.append((fname, bbs, scs, alls))
        except Exception as e:
            print("[FAIL] {} - {}".format(fname, e))

    with open(os.path.join(output_folder, "rmsd_report.txt"), "w") as f:
        for fname, bb, sc, all_atoms in report_rows:
            f.write("{}\tBackbone:{}\tSidechain:{}\tAll:{}\n".format(fname, bb, sc, all_atoms))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 adding_sidechains.py <input_folder>")
        exit(1)
    main(sys.argv[1])
