"""Dataset and model registry for the LLM MoleculeNet evaluation.

Each dataset entry lists its label columns (as written by prepare_data.py) with a
plain-English description, and the quantity/units the model is asked to predict.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets'
RUNS_DIR = ROOT / 'runs'

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
            'FDA_APPROVED': 'Is an FDA-approved drug.',
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

DEFAULT_MODELS = ['claude-haiku-4-5', 'gpt-5-mini']
DEFAULT_SHOTS = [0, 5, 20]

# Per-model request settings. Unknown models get no extra settings.
#   temperature      - only sent when set (newer Claude and GPT-5 models reject it)
#   effort           - Anthropic output_config.effort (models with adaptive thinking)
#   reasoning_effort - OpenAI reasoning models
#   thinking_headroom - extra max_tokens reserved for reasoning; unused tokens are not billed
MODEL_SETTINGS = {
    'claude-haiku-4-5': {'temperature': 0.0},
    'claude-sonnet-5-5': {'effort': 'low', 'thinking_headroom': 4096},
    'claude-opus-5-5': {'effort': 'low', 'thinking_headroom': 4096},
    'gpt-5-nano': {'reasoning_effort': 'minimal', 'thinking_headroom': 2048},
    'gpt-5-mini': {'reasoning_effort': 'minimal', 'thinking_headroom': 2048},
    'gpt-5': {'reasoning_effort': 'minimal', 'thinking_headroom': 2048},
    'gpt-4.1-mini': {'temperature': 0.0},
    'gpt-4.1': {'temperature': 0.0},
}

# Prices (USD per token) registered with litellm when its cost map lacks the model.
PRICE_FALLBACK = {
    'claude-sonnet-5-5': {'input_cost_per_token': 2e-6, 'output_cost_per_token': 1e-5,
                          'cache_read_input_token_cost': 2e-7, 'cache_creation_input_token_cost': 2.5e-6,
                          'litellm_provider': 'anthropic', 'mode': 'chat'},
    'claude-opus-5-5': {'input_cost_per_token': 4e-6, 'output_cost_per_token': 2e-5,
                        'cache_read_input_token_cost': 2e-7, 'cache_creation_input_token_cost': 5e-6,
                        'litellm_provider': 'anthropic', 'mode': 'chat'},
}

# Minimum prompt-prefix length (tokens) Anthropic will cache; shorter prefixes are
# not marked with cache_control since a marker there only risks a cache-write premium.
# Models not listed use 1024.
ANTHROPIC_CACHE_MIN_TOKENS = {'claude-haiku-4-5': 4096, 'claude-sonnet-5-5': 512, 'claude-opus-5-5': 512}

BATCH_DISCOUNT = 0.5
