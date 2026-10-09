"""Dataset registry for the LLM MoleculeNet evaluation, shared by Claude_GPT_eval and DeepRetro.

Each dataset entry lists its label columns (as written by prepare_data.py) with a
plain-English description, and the quantity/units the model is asked to predict.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets'

CLASSIFICATION_UNIT = 'probability between 0 and 1 (unitless) that the statement is true'

TOX21_ASSAY = 'Is active in the Tox21 quantitative high-throughput screening assay for'
SIDER_PREFIX = 'Has marketed-drug adverse reactions (SIDER database) in the MedDRA system organ class'

DATASETS = {
    # ---------------- classification ----------------
    'bbbp': {
        'type': 'classification',
        'description': 'BBBP - blood-brain barrier penetration of compounds.',
        'tasks': {'p_np': 'Penetrates the blood-brain barrier (BBB permeable).'},
    },
    'bace_classification': {
        'type': 'classification',
        'description': 'BACE - inhibitors of human beta-secretase 1 (BACE-1).',
        'tasks': {'Class': 'Is a potent inhibitor of human BACE-1 (roughly IC50 <= 100 nM, i.e. pIC50 >= 7).'},
    },
    'clintox': {
        'type': 'classification',
        'description': 'ClinTox - FDA-approved drugs and drug candidates that failed clinical trials for toxicity.',
        'tasks': {
            # 'FDA_APPROVED': 'Is an FDA-approved drug.',
            'CT_TOX': 'Failed clinical trials for toxicity reasons.',
        },
    },
    'hiv': {
        'type': 'classification',
        'description': 'HIV - NCI/DTP AIDS antiviral screen of compounds for inhibition of HIV replication.',
        'tasks': {'HIV_active': 'Is confirmed active or moderately active (CA/CM) at inhibiting HIV replication.'},
    },
    'tox21': {
        'type': 'classification',
        'description': 'Tox21 - toxicity of compounds measured in nuclear-receptor and stress-response assays.',
        'tasks': {
            'NR-AR': f'{TOX21_ASSAY} androgen receptor agonism (full-length receptor).',
            'NR-AR-LBD': f'{TOX21_ASSAY} androgen receptor ligand-binding-domain agonism.',
            'NR-AhR': f'{TOX21_ASSAY} aryl hydrocarbon receptor agonism.',
            'NR-Aromatase': f'{TOX21_ASSAY} aromatase (CYP19A1) inhibition.',
            'NR-ER': f'{TOX21_ASSAY} estrogen receptor alpha agonism (full-length receptor).',
            'NR-ER-LBD': f'{TOX21_ASSAY} estrogen receptor alpha ligand-binding-domain agonism.',
            'NR-PPAR-gamma': f'{TOX21_ASSAY} PPAR-gamma agonism.',
            'SR-ARE': f'{TOX21_ASSAY} antioxidant response element (Nrf2/ARE) activation.',
            'SR-ATAD5': f'{TOX21_ASSAY} ATAD5 induction (genotoxic stress).',
            'SR-HSE': f'{TOX21_ASSAY} heat shock factor response element activation.',
            'SR-MMP': f'{TOX21_ASSAY} mitochondrial membrane potential disruption.',
            'SR-p53': f'{TOX21_ASSAY} p53 pathway activation (DNA damage response).',
        },
    },
    'sider': {
        'type': 'classification',
        'description': 'SIDER - side effects of marketed drugs grouped by MedDRA system organ class.',
        'tasks': {col: f"{SIDER_PREFIX} '{col}'." for col in [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders', 'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions',
            'Endocrine disorders', 'Surgical and medical procedures',
            'Vascular disorders', 'Blood and lymphatic system disorders',
            'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders', 'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
            'Renal and urinary disorders',
            'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders',
            'Nervous system disorders', 'Injury, poisoning and procedural complications',
        ]},
    },
    # ---------------- regression ----------------
    'delaney': {
        'type': 'regression',
        'description': 'ESOL (Delaney) - measured aqueous solubility of compounds at room temperature.',
        'tasks': {'measured log solubility in mols per litre': 'Aqueous solubility, log10 of the molar solubility (logS).'},
        'unit': 'log10(mol/L)',
    },
    'freesolv': {
        'type': 'regression',
        'description': 'FreeSolv - experimental hydration free energies of small molecules in water.',
        'tasks': {'y': 'Hydration free energy (free energy of transfer from gas phase to water).'},
        'unit': 'kcal/mol',
    },
    'lipo': {
        'type': 'regression',
        'description': 'Lipophilicity - AstraZeneca octanol/water distribution coefficients measured at pH 7.4.',
        'tasks': {'exp': 'Octanol/water distribution coefficient at pH 7.4 (logD7.4).'},
        'unit': 'log10 units (logD, dimensionless)',
    },
    'bace_regression': {
        'type': 'regression',
        'description': 'BACE - binding affinity of inhibitors of human beta-secretase 1 (BACE-1).',
        'tasks': {'pIC50': 'pIC50 against human BACE-1, i.e. -log10 of the IC50 expressed in mol/L.'},
        'unit': 'pIC50 units (-log10 M)',
    },
    'clearance': {
        'type': 'regression',
        'description': 'Clearance - AstraZeneca in vitro intrinsic clearance measured in human liver microsomes.',
        'tasks': {'target': 'Intrinsic clearance in human liver microsomes (CLint).'},
        'unit': 'uL/min/mg protein',
    },
}

DEFAULT_SHOTS = [0, 5, 20]
