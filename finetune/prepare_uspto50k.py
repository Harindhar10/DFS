"""Build LLaMA-Factory SFT data for one-step retrosynthesis from DeepChem's USPTO-50K."""
import argparse, json, os, random

import pandas as pd
from rdkit import Chem, RDLogger
from deepchem.utils.data_utils import download_url, get_data_dir

RDLogger.DisableLog("rdApp.*")
USPTO_50K_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_50K.csv"

SYSTEM = "You are an expert organic chemist specializing in single-step retrosynthesis."
INSTRUCTION = ("Predict the reactants needed to synthesize the given product. "
               "Answer with only the reactant SMILES, separated by '.'.")


def canon(smi):
    """Strip atom maps and canonicalize; None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def canon_multi(smi):
    parts = [canon(s) for s in smi.split(".") if s]
    return None if not parts or any(p is None for p in parts) else ".".join(sorted(parts))


def reaction_column(df):
    for col in df.columns:
        if df[col].astype(str).str.contains(">").mean() > 0.9:
            return col
    raise ValueError(f"no reaction-SMILES column in {df.columns.tolist()}")


def to_record(prod, reac):
    return {"system": SYSTEM, "instruction": INSTRUCTION, "input": prod, "output": reac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="finetune/data")
    ap.add_argument("--exclude_csv", default="DeepRetro/data/uspto_50k_test_250.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = get_data_dir()
    csv_path = os.path.join(data_dir, "USPTO_50K.csv")
    if not os.path.exists(csv_path):
        download_url(USPTO_50K_URL, dest_dir=data_dir)
    raw = pd.read_csv(csv_path)
    col = reaction_column(raw)
    print(f"{len(raw)} rows, reaction column = {col!r}")

    rows, seen = [], set()
    for rxn in raw[col].astype(str):
        fields = rxn.split(">")          # reactants>reagents>product (reagents may be empty)
        reac, prod = canon_multi(fields[0]), canon_multi(fields[-1])
        if reac is None or prod is None or "." in prod or (prod, reac) in seen:
            continue
        seen.add((prod, reac))
        rows.append((prod, reac))
    print(f"{len(rows)} clean single-product reactions")

    # Remove every product in the notebook's eval set, so the benchmark measures generalization.
    excl = {canon_multi(s) for s in pd.read_csv(args.exclude_csv)["input"].astype(str)}
    before = len(rows)
    rows = [r for r in rows if r[0] not in excl]
    print(f"dropped {before - len(rows)} reactions whose product is in {args.exclude_csv}")

    # Split by product, so the same product never appears in two splits.
    products = sorted({p for p, _ in rows})
    random.Random(args.seed).shuffle(products)
    n = len(products)
    split_of = {p: ("train" if i < 0.9 * n else "val" if i < 0.95 * n else "test")
                for i, p in enumerate(products)}

    os.makedirs(args.out_dir, exist_ok=True)
    info = {}
    for split in ("train", "val", "test"):
        recs = [to_record(p, r) for p, r in rows if split_of[p] == split]
        fname = f"uspto50k_retro_{split}.json"
        with open(os.path.join(args.out_dir, fname), "w") as f:
            json.dump(recs, f)
        info[f"uspto50k_retro_{split}"] = {
            "file_name": fname,
            "columns": {"system": "system", "prompt": "instruction",
                        "query": "input", "response": "output"},
        }
        print(f"{split}: {len(recs)}")

    # The 250-molecule notebook set, in the same format, for a direct comparison.
    bench = pd.read_csv(args.exclude_csv)
    recs = [to_record(canon_multi(str(i)), canon_multi(str(o)))
            for i, o in zip(bench["input"], bench["output"])]
    with open(os.path.join(args.out_dir, "uspto50k_bench250.json"), "w") as f:
        json.dump(recs, f)
    info["uspto50k_bench250"] = dict(info["uspto50k_retro_test"], file_name="uspto50k_bench250.json")

    with open(os.path.join(args.out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
