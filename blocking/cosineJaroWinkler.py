import pandas as pd
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from connectors import devo
import os
import time
import torch
from transformers import AutoTokenizer
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

MODEL_NAME = 'distilbert-base-uncased'
TOP_K = 10
MIN_COSINE = 0.60
MIN_JARO = 0.85

output_folder = 'blocking'
if not os.path.exists(output_folder): os.makedirs(output_folder)
print(f"Loading Neural Model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

today = datetime.today().strftime('%Y-%m-%d')

# =======================================================================
#     DistilBERT tokenizer + Cosine Similarity + Jaro-Winkler distance
# =======================================================================

def get_distilbert_indicators(df, text_col, batch_size=20000):
    texts = df[text_col].astype(str).tolist()
    total_records = len(texts)
    
    data = []
    row = []
    col = []
    
    vocab_size = tokenizer.vocab_size

    encodings = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=512)

    print("   Building sparse matrix...")
    for i, token_ids in enumerate(encodings['input_ids']):
        unique_tokens = sorted(list(set(token_ids)))
        
        if not unique_tokens: continue
        
        n_tokens = len(unique_tokens)
        
        data.extend([1] * n_tokens)       # Exists = 1
        row.extend([i] * n_tokens)        # Row i (actual entity)
        col.extend(unique_tokens)         # Column (DistilBERT token ID)

    # Create float matrix for cosine similarity
    indicators = csr_matrix((data, (row, col)), shape=(total_records, vocab_size), dtype=np.float32)
    return indicators

def run_smart_blocking(group_df, text_col='blocking_text', top_k=10, min_cosine_score=0.4, jaro_threshold=0.85):
    """
    1. Generate candidates using DistilBERT tokenizer (matrix) + Cosine Similarity
    2. Filter candidates using Jaro-Winkler distance (String Similarity)
    """
    if len(group_df) < 2: return pd.DataFrame()

    # Generate and normalize matrix
    indicators = get_distilbert_indicators(group_df, text_col)
    indicators_norm = normalize(indicators, norm='l2', axis=1)
    
    results = []
    n_records = indicators_norm.shape[0]

    iterator = range(n_records)
    if n_records > 1000: iterator = tqdm(iterator, desc="Matching & Filtering", leave=False)

    for i in iterator:
        # Inner Product -> normalized vectors = cosine
        cosine_vector = indicators_norm[i, :].dot(indicators_norm.transpose())
        cosine_scores = cosine_vector.toarray()[0]
        
        # First candidate filter (Cosine Similarity)
        candidate_index = np.where(cosine_scores >= min_cosine_score)[0]
        candidate_index = candidate_index[candidate_index > i] # Solo triángulo superior
        
        if len(candidate_index) == 0: continue

        # Use MIN_COSINE cosine threshold
        if len(candidate_index) > top_k * 2:
             top_k_index = np.argpartition(cosine_scores[candidate_index], -top_k*2)[-top_k*2:]
             best_index = candidate_index[top_k_index]
        else:
            best_index = candidate_index

        # Second candidate filter (Jaro-Winkler)
        lid_val = group_df.iloc[i]['entty_riad_cd']
        lname = group_df.iloc[i]['nm_entty']
        
        for neighbor_idx in best_index:
            rname = group_df.iloc[neighbor_idx]['nm_entty']
            jw_score = 1.0 - JaroWinkler.distance(lname, rname)
            
            # Use MIN_JARO cosine threshold
            if jw_score < jaro_threshold:
                continue

            rid_val = group_df.iloc[neighbor_idx]['entty_riad_cd']
            
            results.append({
                'entty_riad_cd': lid_val,
                'matched_entty_riad_cd': rid_val,
                'nm_entty_1': lname,
                'nm_entty_2': rname,
                'cntry': group_df.iloc[i]['cntry'],
                'cosine_score': round(cosine_scores[neighbor_idx], 3),
                'similarity_score': round(jw_score, 3)
            })

    return pd.DataFrame(results)

print("\nFetching data from DEVO...")
query = f'''
SELECT entty_riad_cd, cntry, nm_entty, cty 
FROM crp_riad.riad_entty_d_1 
WHERE vld_frm <= '{today}' AND vld_t >= '{today}'
AND entty_riad_cd IS NOT NULL
AND cntry NOT IN ('__', 'AT','BE','BG','CY','CZ','DE','DK', 'EE', 'ES', 'FI','FR', 'GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK')
'''
df_all = devo.read_sql(query).fillna('')
df_all['blocking_text'] = df_all['nm_entty']

all_results = []
countries = df_all['cntry'].unique()

print(f"Starting Smart Blocking + Jaro Filter...")
total_start = time.time()

for country in countries:
    country_df = df_all[df_all['cntry'] == country].copy().reset_index(drop=True)
    
    print(f"Processing {country} ({len(country_df)} entities)...")
    t0 = time.time()

    matches = run_smart_blocking(country_df, 
                                 top_k=TOP_K,
                                 min_cosine_score=MIN_COSINE, 
                                 jaro_threshold=MIN_JARO) 
    t1 = time.time()
    elapsed = t1 - t0
    
    if not matches.empty:
        print(f"   Saved {len(matches)} matches.")
        print(f"   > Found {len(matches)} candidates in {elapsed:.2f}s ({len(matches)/elapsed:.0f} pairs/sec)")
        all_results.append(matches)

if all_results:
    final_df = pd.concat(all_results)
    print(f"Total Candidate Pairs Found: {len(final_df)} in {time.time() - total_start:.1f}s")
    
    path = os.path.join(output_folder, f"smart_blocking_results_{today}.csv")
    final_df.to_csv(path, index=False)
    print(f"Saved to {path}")