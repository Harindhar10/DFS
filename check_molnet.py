"""Find rows in a CSV whose SMILES matches any of the given molecules,
however each SMILES is written.

Usage: python find_molecule.py --file your_file.csv --molecules "SMILES1" "SMILES2" ... [--ignore-stereo]
Example: python find_molecule.py --file data.csv --molecules "CC(=O)Oc1ccccc1C(=O)O" "CCc1ccccc1"
Requires: pip install rdkit pandas
"""
import argparse
import sys

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silence warnings for unparseable SMILES


def canonical(smi, ignore_stereo=False):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    # isomericSmiles=False drops @/@@ and /\ marks (and isotope labels)
    return Chem.MolToSmiles(mol, isomericSmiles=not ignore_stereo)


def main(path, queries, ignore_stereo=False):
    targets = {}
    for q in queries:
        c = canonical(q, ignore_stereo)
        if c is None:
            sys.exit(f"Could not parse query SMILES: {q}")
        targets[q] = c

    df = pd.read_csv(path)
    canon = df["smiles"].map(lambda s: canonical(s, ignore_stereo))

    bad = df.index[canon.isna()].tolist()
    if bad:
        print(f"Skipped {len(bad)} unparseable SMILES at indexes: {bad}")

    results = {}
    for q, target in targets.items():
        hits = df.index[canon == target].tolist()
        results[q] = hits
        print(f"\n{q}: matches at indexes {hits}")
        if hits:
            print(df.loc[hits].to_string())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find rows in a CSV whose SMILES matches any of the given molecules."
    )
    parser.add_argument("--file", required=True, help="path to the CSV file (needs a 'smiles' column)")
    parser.add_argument(
        "--molecules", required=True, nargs="+",
        help="one or more SMILES of molecules to search for, separated by spaces",
    )
    parser.add_argument(
        "--ignore-stereo", action="store_true",
        help="ignore stereochemistry (and isotopes) when comparing molecules",
    )
    args = parser.parse_args()
    main(args.file, args.molecules, args.ignore_stereo)