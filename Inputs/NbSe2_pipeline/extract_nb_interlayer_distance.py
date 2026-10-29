#!/usr/bin/env python3
"""Extract Nb layer heights from the original Wannier90 logs (Python standard library).

Run in grid12:
    python3 extract_nb_interlayer_distance.py
Or specify its location:
    python3 extract_nb_interlayer_distance.py /path/to/grid12 --output NbSe2_interlayer_distance.dat

The result is whitespace-delimited, with comment lines starting with #.
distance_A = abs(z_Nb2 - z_Nb1), using the Cartesian atomic Site table.
No Wannier centres, lateral displacement, or periodic wrapping enter this distance.
"""

import argparse
import math
from pathlib import Path
import re
import sys


def number(value):
    return float(value.replace("D", "E").replace("d", "e"))


def read_geometry(path):
    text = path.read_text()
    lattice_block = re.search(
        r"Lattice Vectors \(Ang\)(.*?)Unit Cell Volume", text, re.S
    )
    if lattice_block is None:
        raise ValueError("missing lattice vectors in Angstrom")
    lattice = {}
    for line in lattice_block[1].splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] in ("a_1", "a_2", "a_3"):
            lattice[fields[0]] = tuple(map(number, fields[1:]))
    if len(lattice) != 3:
        raise ValueError("expected three lattice vectors")

    lines = text.splitlines()
    start = next((k for k, line in enumerate(lines)
                  if "Site" in line and "Cartesian Coordinate (Ang)" in line), None)
    if start is None:
        raise ValueError("missing atomic Site table; reconstructed .wout files cannot be used")
    nb = {}
    for line in lines[start + 1:]:
        if line.strip().startswith("*---"):
            break
        fields = line.split("|")
        if len(fields) < 4:
            continue
        site = fields[1].split()
        if not site or site[0] != "Nb":
            continue
        cart = fields[2].split()
        if len(site) != 5 or len(cart) != 3:
            raise ValueError("malformed Nb atom row")
        index = int(site[1])
        if index in nb:
            raise ValueError("duplicate Nb atom index")
        nb[index] = tuple(map(number, cart))
    if len(nb) != 2:
        raise ValueError("expected exactly two Nb atoms, found {}".format(len(nb)))
    if not all(math.isfinite(v) for xyz in list(lattice.values()) + list(nb.values()) for v in xyz):
        raise ValueError("non-finite atomic coordinates or lattice vectors")
    if any(abs(lattice[key][2]) > 1e-6 for key in ("a_1", "a_2")):
        raise ValueError("a1/a2 are not in the xy plane; z is not the layer normal")
    z_bottom, z_top = sorted(xyz[2] for xyz in nb.values())
    return lattice, z_bottom, z_top


def extract(root, ngrid):
    rows, errors = [], []
    reference = None
    for i in range(ngrid):
        for j in range(ngrid):
            path = root / "shift_i{:02d}_j{:02d}".format(i, j) / "NbSe2.wout"
            try:
                lattice, z_bottom, z_top = read_geometry(path)
                if reference is None:
                    reference = lattice
                if any(abs(lattice[key][k] - reference[key][k]) > 1e-5
                       for key in reference for k in range(3)):
                    raise ValueError("lattice differs from the first stacking")
                u, v = i / ngrid, j / ngrid
                dx, dy = (u * lattice["a_1"][k] + v * lattice["a_2"][k] for k in (0, 1))
                rows.append((i, j, u, v, dx, dy, z_bottom, z_top, z_top - z_bottom))
            except (OSError, ValueError) as exc:
                errors.append("{}: {}".format(path, exc))
    if errors:
        raise ValueError("No output written: {} of {} stackings failed.\n{}".format(
            len(errors), ngrid * ngrid, "\n".join(errors)))
    return reference, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="grid12 directory")
    parser.add_argument("--output", type=Path, default=Path("NbSe2_interlayer_distance.dat"))
    parser.add_argument("--ngrid", type=int, default=12)
    args = parser.parse_args()
    if args.ngrid < 1:
        parser.error("--ngrid must be positive")
    try:
        lattice, rows = extract(args.root, args.ngrid)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with args.output.open("w") as out:
        out.write("# Nb atomic layer separation from original NbSe2.wout Site tables\n")
        out.write("# source_grid: {}\n".format(args.root.resolve()))
        out.write("# ngrid: {}\n".format(args.ngrid))
        out.write("# d = u*a1 + v*a2; u=i/ngrid, v=j/ngrid\n")
        out.write("# distance_A = abs(z_Nb2-z_Nb1); no lateral distance or periodic wrapping\n")
        for key in ("a_1", "a_2", "a_3"):
            out.write("# {}_Ang: {:.8f} {:.8f} {:.8f}\n".format(key, *lattice[key]))
        out.write("# i j u v dx_A dy_A z_bottom_A z_top_A distance_A\n")
        for row in rows:
            out.write("{:d} {:d} {:.10f} {:.10f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(*row))
    print("Wrote {} stackings to {}".format(len(rows), args.output))
    print("Nb-Nb vertical distance: {:.5f} to {:.5f} Angstrom".format(
        min(row[-1] for row in rows), max(row[-1] for row in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
