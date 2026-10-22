"""Build LLaMA-Factory SFT data for MoleculeNet property prediction from the DeepChem scaffold
splits in moleculenet/datasets/<name>/{train,valid,test}.csv (written by moleculenet/prepare_data.py).

One example per labeled (molecule, task) cell; NaN cells (missing multi-task labels) are skipped.
Each dataset gets its own train/val/test files so one adapter can be trained per dataset.
"""
import argparse, json, os

import pandas as pd

SYSTEM = "You are an expert medicinal chemist who predicts molecular properties from SMILES."
CLS_SUFFIX = " Answer with only Yes or No."
REG_SUFFIX = " Answer with only the number."

TOX21 = {
    "NR-AR": "the androgen receptor (NR-AR)",
    "NR-AR-LBD": "the androgen receptor ligand-binding domain (NR-AR-LBD)",
    "NR-AhR": "the aryl hydrocarbon receptor (NR-AhR)",
    "NR-Aromatase": "the aromatase enzyme (NR-Aromatase)",
    "NR-ER": "the estrogen receptor alpha (NR-ER)",
    "NR-ER-LBD": "the estrogen receptor ligand-binding domain (NR-ER-LBD)",
    "NR-PPAR-gamma": "the peroxisome proliferator-activated receptor gamma (NR-PPAR-gamma)",
    "SR-ARE": "the antioxidant response element (SR-ARE)",
    "SR-ATAD5": "the ATAD5 genotoxicity (SR-ATAD5)",
    "SR-HSE": "the heat shock factor response element (SR-HSE)",
    "SR-MMP": "the mitochondrial membrane potential (SR-MMP)",
    "SR-p53": "the p53 DNA damage response (SR-p53)",
}

# name -> (kind, {csv column: question}). SIDER questions are built from the column names below.
DATASETS = {
    "bbbp": ("cls", {"p_np": "Does this molecule penetrate the blood-brain barrier?"}),
    "bace_classification": ("cls", {"Class": "Is this molecule an inhibitor of human beta-secretase 1 (BACE-1)?"}),
    "hiv": ("cls", {"HIV_active": "Does this molecule inhibit HIV replication?"}),
    "clintox": ("cls", {
        "FDA_APPROVED": "Is this molecule an FDA-approved drug?",
        "CT_TOX": "Did this molecule fail clinical trials for toxicity reasons?",
    }),
    "tox21": ("cls", {k: f"Is this molecule active in the Tox21 assay for {v}?" for k, v in TOX21.items()}),
    "sider": ("cls", {}),
    "delaney": ("reg", {"measured log solubility in mols per litre":
                        "Predict the aqueous solubility of this molecule as log10(mol/L)."}),
    "freesolv": ("reg", {"y": "Predict the hydration free energy of this molecule (FreeSolv, as given in the DeepChem split)."}),
    "lipo": ("reg", {"exp": "Predict the octanol/water distribution coefficient (logD at pH 7.4) of this molecule."}),
    "bace_regression": ("reg", {"pIC50": "Predict the pIC50 of this molecule against human beta-secretase 1 (BACE-1)."}),
    "clearance": ("reg", {"target": "Predict the metabolic clearance of this molecule."}),
}

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # our name -> CSV file stem
COLUMNS = {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}


def questions(name, df):
    kind, qs = DATASETS[name]
    if name == "sider":
        qs = {c: f"Does this drug cause side effects in the category '{c}'?"
              for c in df.columns if c != "smiles"}
    missing = [c for c in qs if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: columns {missing} not in CSV {df.columns.tolist()}")
    return kind, qs


def to_records(df, kind, qs, with_meta):
    recs = []
    for task, q in qs.items():
        suffix = CLS_SUFFIX if kind == "cls" else REG_SUFFIX
        for smi, y in zip(df["smiles"], df[task]):
            if pd.isna(y):
                continue
            out = ("Yes" if y >= 0.5 else "No") if kind == "cls" else f"{y:.2f}"
            rec = {"system": SYSTEM, "instruction": q + suffix, "input": smi, "output": out}
            if with_meta:  # read by score_moleculenet.py; LLaMA-Factory ignores unmapped fields
                rec.update(task=task, label=float(y))
            recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="moleculenet/datasets")
    ap.add_argument("--out_dir", default="moleculenet/data")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    info_path = os.path.join(args.out_dir, "dataset_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}  # --datasets subset keeps the rest
    for name in args.datasets.split(","):
        for split, stem in SPLITS.items():
            df = pd.read_csv(os.path.join(args.splits_dir, name, f"{stem}.csv"))
            kind, qs = questions(name, df)
            recs = to_records(df, kind, qs, with_meta=True)
            fname = f"{name}_{split}.json"
            with open(os.path.join(args.out_dir, fname), "w") as f:
                json.dump(recs, f)
            info[f"{name}_{split}"] = {"file_name": fname, "columns": COLUMNS}

            stat = (f"pos={sum(r['output'] == 'Yes' for r in recs) / max(len(recs), 1):.3f}"
                    if kind == "cls" else "")
            print(f"{name:20s} {split:5s} mols={len(df):6d} tasks={len(qs):2d} examples={len(recs):6d} {stat}")

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
