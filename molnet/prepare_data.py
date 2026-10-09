"""Write DeepChem scaffold splits of the MoleculeNet datasets as CSVs.

    python -m molnet.prepare_data                   # all datasets in config.DATASETS
    python -m molnet.prepare_data --datasets bbbp,hiv

Each dataset becomes <data_dir>/<name>/{train,valid,test}.csv with a `smiles` column and
one column per task. Molecules with SMILES longer than 200 characters are dropped.
"""
import argparse
from pathlib import Path

import deepchem as dc
import pandas as pd

from .config import DATA_DIR, DATASETS

MAX_SMILES_LEN = 200


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', default=','.join(DATASETS), help='comma-separated MoleculeNet dataset names')
    p.add_argument('--data_dir', default=str(DATA_DIR), help='output directory')
    args = p.parse_args()

    for name in args.datasets.split(','):
        # DummyFeaturizer keeps the raw SMILES strings as X.
        tasks, splits, _ = getattr(dc.molnet, f'load_{name}')(
            featurizer=dc.feat.DummyFeaturizer(), transformers=[], splitter='scaffold', reload=False)
        out_dir = Path(args.data_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, ds in zip(['train', 'valid', 'test'], splits):
            df = pd.DataFrame({'smiles': ds.X, **{t: ds.y[:, i] for i, t in enumerate(tasks)}})
            keep = df['smiles'].str.len() <= MAX_SMILES_LEN
            df[keep].to_csv(out_dir / f'{split}.csv', index=False)
            print(f'{name} {split}: {keep.sum()} molecules ({(~keep).sum()} dropped, SMILES > {MAX_SMILES_LEN})')


if __name__ == '__main__':
    main()
