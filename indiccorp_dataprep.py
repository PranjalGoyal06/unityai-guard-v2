from datasets import load_dataset
import pandas as pd
import re

HF_DATASET = "ai4bharat/IndicCorpV2"
OUTPUT_CSV = "safe_indiccorp_subset.csv"

lang_targets = {
    "ben_Beng": 10000,  # Bengali
    "kan_Knda": 20000,  # Kannada
    "mal_Mlym": 20000,  # Malayalam
    "ory_Orya": 30000   # Odia
}

# Cleaning helper:
def basic_clean(text):
    if not text or len(text.strip()) < 20:
        return None
    if len(re.findall(r"[!@#$%^&*_=+<>/\\|~`{}\[\]]", text)) > 5:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text


all_data = []

# Process each language:
for lang_code, target_count in lang_targets.items():
    print(f"\nProcessing {lang_code}...")
    
    ds = load_dataset(HF_DATASET, split=lang_code, streaming=True)
    
    cleaned = []
    seen = set()
    count = 0
    
    for item in ds:
        if count >= target_count * 100: # safety cap
            break
        
        t = basic_clean(item["text"])
        if not t or t in seen:
            count += 1
            continue
        
        seen.add(t)
        cleaned.append({"text": t, "lang": lang_code, "label": "safe"})
        count += 1
        
        if len(cleaned) >= target_count:
            break
    
    print(f"Cleaned samples retained: {len(cleaned)}")
    all_data.extend(cleaned)

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_CSV, index=False)
print("\nSaved cleaned subset to ", OUTPUT_CSV)
print(df.groupby('lang').size())