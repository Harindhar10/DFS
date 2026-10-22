"""Top-1 exact match on LLaMA-Factory generated_predictions.jsonl (all_correct rule from code_v1.ipynb)."""
import json, sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon_set(smi):
    out = []
    for s in smi.strip().split("."):
        try:
            out.append(Chem.CanonSmiles(s))
        except Exception:
            return None
    return sorted(out)


n = hit = invalid = 0
with open(sys.argv[1]) as f:
    for line in f:
        rec = json.loads(line)
        pred = rec["predict"].strip().split()[0] if rec["predict"].strip() else ""
        gt, pr = canon_set(rec["label"]), canon_set(pred)
        n += 1
        invalid += pr is None
        hit += pr is not None and pr == gt
print(f"n={n}  top-1 exact match={100 * hit / n:.2f}%  invalid SMILES={100 * invalid / n:.2f}%")
