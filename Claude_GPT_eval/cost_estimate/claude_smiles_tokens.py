import pandas as pd
import Claude_GPT_eval.cost_estimate.moleculenet_cost_anthropic as mc

total_chars, total_tokens = 0, 0
for name in mc.DATASETS:
    df = pd.read_csv(f"Claude_GPT_eval/datasets/{name}/train.csv")
    sample = df["smiles"].sample(min(300, len(df)), random_state=0).tolist()
    cpt = mc.calibrate_chars_per_token(sample)
    chars = len("\n".join(sample))
    total_chars += chars
    total_tokens += chars / cpt
    print(f"{name:<20} {cpt:.2f}")

mc.CHARS_PER_TOKEN = total_chars / total_tokens   # pooled across all datasets
print("pooled:", round(mc.CHARS_PER_TOKEN, 2))