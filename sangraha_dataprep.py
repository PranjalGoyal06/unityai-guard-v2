from datasets import load_dataset
import pandas as pd
import re
import random

# target language configs
lang_targets = {
    "mal": 20000,  # Malayalam
    "kan": 20000,  # Kannada
    "ory": 30000,  # Odia
    "ben": 10000   # Bengali
}

# load dataset
print("Loading Sangraha dataset...")
ds = load_dataset("ai4bharat/sangraha", split="train")

# quick cleaning helper
def basic_clean(text):
    if not text or len(text.strip()) < 20:
        return None
    if len(re.findall(r"[!@#$%^&*_=+<>/\\|~`{}\[\]]", text)) > 5:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text

all_data = []

for lang_code, target_count in lang_targets.items():
    print(f"\nProcessing {lang_code}...")
    sub = ds.filter(lambda x: x["lang"] == lang_code)
    print(f"  Found {len(sub)} samples")

    # shuffle and sample roughly target_count (skip if fewer)
    n = min(target_count, len(sub))
    sample = sub.shuffle(seed=42).select(range(n))

    cleaned = []
    seen = set()
    for t in sample["text"]:
        t = basic_clean(t)
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append({"text": t, "lang": lang_code, "label": "safe"})

    print(f"  Cleaned samples retained: {len(cleaned)}")
    all_data.extend(cleaned)

# combine and save
df = pd.DataFrame(all_data)
df.to_csv("safe_sangraha_subset.csv", index=False)
print("\nSaved cleaned subset to safe_sangraha_subset.csv")
print(df.groupby('lang').size())