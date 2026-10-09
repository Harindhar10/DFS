"""Model registry for the Claude/GPT MoleculeNet evaluation (datasets live in molnet.config)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / 'runs'

DEFAULT_MODELS = ['claude-haiku-4-5', 'gpt-5-mini']

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
