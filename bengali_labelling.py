# direct_label_mapping.py
# Requirements: pandas
# Run: python direct_label_mapping.py
# Edit file paths if your CSVs are named differently.

import pandas as pd

# ---------- Config: file paths ----------
COMMENTS_CSV = "raw_data/bengali_comments.csv"    # expects columns like: text,vulgar,hate,religious,threat,troll,Insult
BATD_AGG_CSV = "raw_data/bengali_aggressive.csv"  # expects columns like: cleaned,Class,Label (all harmful)
BATD_NONAGG_CSV = "raw_data/bengali_non-aggressive.csv"  # expects columns like: text (all safe)

OUT_CSV = "bengali_labelled.csv"

# ---------- S keys ----------
S_KEYS = [
 'S1','S2','S3','S4','S5','S6','S7','S8','S9',
 'S10','S11','S12','S13','S14','S15','S16','S17','S18'
]

# ---------- helpers ----------
def safe_int_flag(x):
    """Return 1 if x represents 1, otherwise 0."""
    try:
        return 1 if int(x) == 1 else 0
    except Exception:
        # sometimes values are '0'/'1' as strings, or NaN
        if isinstance(x, str) and x.strip() == '1':
            return 1
        return 0

def ensure_S_columns(df):
    for s in S_KEYS:
        if s not in df.columns:
            df[s] = 0
    return df

# ---------- load ----------
comm = pd.read_csv(COMMENTS_CSV, encoding="utf-8", encoding_errors="ignore")
agg = pd.read_csv(BATD_AGG_CSV, encoding="utf-8", encoding_errors="ignore")
nonagg = pd.read_csv(BATD_NONAGG_CSV, encoding="utf-8", encoding_errors="ignore")

# unify text column name
if 'text' in comm.columns:
    comm = comm.rename(columns={'text':'text_raw'})
if 'cleaned' in agg.columns:
    agg = agg.rename(columns={'cleaned':'text_raw'})
if 'text' in nonagg.columns:
    nonagg = nonagg.rename(columns={'text':'text_raw'})

# ensure S columns exist
comm = ensure_S_columns(comm)
agg = ensure_S_columns(agg)
nonagg = ensure_S_columns(nonagg)

# ---------- DIRECT MAPPINGS (exact as requested) ----------

# Offensive comments mapping:
# - vulgar      -> S12
# - hate        -> S10
# - religious   -> S15 and S10
# - threat      -> S1
# - troll       -> S17
# - Insult      -> S17 and S10
def map_offensive_row(row):
    # vulgar -> S12
    if safe_int_flag(row.get('vulgar', 0)):
        row['S12'] = 1

    # hate -> S10
    if safe_int_flag(row.get('hate', 0)):
        row['S10'] = 1

    # religious -> S15 and S10 (play-safe)
    if safe_int_flag(row.get('religious', 0)):
        row['S15'] = 1
        row['S10'] = 1

    # threat -> S1
    if safe_int_flag(row.get('threat', 0)):
        row['S1'] = 1

    # troll -> S17
    if safe_int_flag(row.get('troll', 0)):
        row['S17'] = 1

    # Insult -> S17 and S10
    if safe_int_flag(row.get('Insult', 0)):
        row['S17'] = 1
        row['S10'] = 1

    return row

comm = comm.apply(map_offensive_row, axis=1)

# Aggressive Text mapping:
# - ReAG -> S15 and S10
# - VeAG -> S17
# - PoAG -> S13
# - GeAG -> S16 and S10
def map_aggressive_row(row):
    cls = str(row.get('Class', '')).strip()
    if cls == 'ReAG':
        row['S15'] = 1
        row['S10'] = 1
    elif cls == 'VeAG':
        row['S17'] = 1  
    elif cls == 'PoAG':
        row['S13'] = 1  
    elif cls == 'GeAG':
        row['S16'] = 1
        row['S10'] = 1
    return row

agg = agg.apply(map_aggressive_row, axis=1)

# nonagg (bengali_non-aggressive.csv) is all safe; ensure text_raw exists
# leave S columns as zeros

# ---------- combine and save ----------
# Add source labels
comm['source'] = 'Multi Labeled Bengali Toxic Comments'
agg['source'] = 'BAD-Bangla-Aggressive-Text-Dataset'
nonagg['source'] = agg['source']

# For batd, if there's a 'Class' or other field we already mapped S labels above.

# Combine datasets
combined = pd.concat([comm, agg, nonagg], ignore_index=True, sort=False)

# Ensure text_raw exists; if not, try other common fields
if 'text_raw' not in combined.columns and 'text' in combined.columns:
    combined = combined.rename(columns={'text':'text_raw'})

# Compute harmful/safe: for batd all harmful, for non-agg all safe, for comments: if all S keys are 0 -> safe
def compute_harm_flag_and_categories(row):
    # determine if any S key is set
    s_values = [int(row.get(s, 0)) for s in S_KEYS]
    any_harm = any(v == 1 for v in s_values)

    # If this row came from BATD, treat as harmful even if S keys weren't set
    source = row.get('source', '')
    if source == 'BAD-Bangla-Aggressive-Text-Dataset':
        harmful = 1  # harmful = 1
    else:
        harmful = 1 if any_harm else 0  # harmful = 1, safe = 0

    # Build harmful categories string: space-separated S labels where value==1
    cats = [s for s, v in zip(S_KEYS, s_values) if v == 1]
    cats_str = ' '.join(cats)

    return pd.Series({'harmful_safe': harmful, 'harmful_categories': cats_str})

computed = combined.apply(compute_harm_flag_and_categories, axis=1)
combined = pd.concat([combined, computed], axis=1)

# Select required four columns in order: raw text, source, harmful/safe, harmful categories
# raw text -> use 'text_raw' if present else try 'text'
if 'text_raw' in combined.columns:
    raw_col = 'text_raw'
elif 'text' in combined.columns:
    raw_col = 'text'
else:
    # fallback to first column
    raw_col = combined.columns[0]

out_df = combined[[raw_col, 'source', 'harmful_safe', 'harmful_categories']].copy()
# set explicit output column names in the requested order
out_df.columns = ['raw text', 'source', 'harmful/safe', 'harmful categories']

# keep UTF-8-SIG so Excel displays Bengali correctly
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# Print a short summary
print("Saved:", OUT_CSV)
print("Sample counts:")
print(out_df['harmful/safe'].value_counts(dropna=False))
