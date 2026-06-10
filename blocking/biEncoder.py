import pandas as pd
import numpy as np
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
import time
import torch
from tqdm.auto import tqdm
from connectors import devo
import os

MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2' 
TOP_K = 10
MIN_SCORE = 0.60

output_folder = 'blocking'
if not os.path.exists(output_folder): os.makedirs(output_folder)
print(f"Loading Neural Model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on: {device.upper()}")

today = datetime.today().strftime('%Y-%m-%d')

# ==========================================
#            BI-ENCODER (FAISS)
# ==========================================
def run_neural_blocking(df, text_col='blocking_text', top_k=10, min_score=0.6):
    """
    Realiza Blocking Semántico usando Vectores Densos y FAISS.
    """
    texts = df[text_col].astype(str).tolist()
    ids = df['entty_riad_cd'].tolist()
    n_records = len(texts)
    
    if n_records < 2: return pd.DataFrame()

    embeddings = model.encode(texts, 
                            batch_size=64, 
                            show_progress_bar=True, 
                            convert_to_numpy=True, 
                            normalize_embeddings=True)

    # --- FAISS ---
    dimension = embeddings.shape[1]
    # IndexFlatIP = Inner Product -> normalized vectors = cosine
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)

    # --- SEARCH ---
    # Look for k nearest neighbours for each vector
    # D = Distances (Scores), I = Index (position in the dataframe)
    D, I = index.search(embeddings, top_k + 1) # +1 including itself

    results = []
    iterator = range(n_records)
    if n_records > 1000:
        iterator = tqdm(range(n_records), desc="   Formatting candidates", leave=False)
    
    for i in iterator:
        query_id = ids[i]
        query_text = texts[i]
        
        # Closest neighbours for entity i
        for j, neighbor_idx in enumerate(I[i]):
            score = float(D[i][j])
            
            # Ignore entity i (autocorrelation)
            if i == neighbor_idx:
                continue
                
            # Blocking threshold
            if score < min_score:
                continue
            
            # Avoid simetric duplicates (A-B y B-A)
            if neighbor_idx <= i:
                continue

            neighbor_id = ids[neighbor_idx]
            neighbor_text = texts[neighbor_idx]
            
            results.append({
                'left_id': query_id,
                'right_id': neighbor_id,
                'left_name': query_text,
                'right_name': neighbor_text,
                'cntry': df.iloc[i]['cntry'],
                'cosine_score': round(score, 4)
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

print(f"\nStarting Neural Blocking Pipeline (Bi-Encoder + FAISS)...")
print(f"Config: Top-K={TOP_K}, Min Score={MIN_SCORE}")

total_start = time.time()

for country in countries:
    country_df = df_all[df_all['cntry'] == country].copy().reset_index(drop=True)
    n_entities = len(country_df)
    
    print(f"\n--- Processing {country} ({n_entities} entities) ---")
    
    if n_entities < 2:
        print("   Skipping (not enough entities)")
        continue

    t0 = time.time()
    matches = run_neural_blocking(country_df, 
                                  top_k=TOP_K, 
                                  min_score=MIN_SCORE)
    t1 = time.time()
    
    elapsed = t1 - t0
    if not matches.empty:
        print(f"   > Found {len(matches)} candidates in {elapsed:.2f}s ({len(matches)/elapsed:.0f} pairs/sec)")
        print("   > Top 3 examples:")
        print(matches[['left_name', 'right_name', 'cosine_score']].head(3).to_string(index=False))
        all_results.append(matches)
    else:
        print(f"   > No matches found above threshold {MIN_SCORE}.")

if all_results:
    final_df = pd.concat(all_results)
    print("\n" + "="*40)
    print(f"FINISHED in {time.time() - total_start:.1f}s")
    print(f"Total Candidate Pairs: {len(final_df)}")
    print("="*40)
    
    path = os.path.join(output_folder, f"neural_blocking_results_{today}.csv")
    final_df.to_csv(path, index=False)
    print(f"Saved to {path}")