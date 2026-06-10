import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import os
from connectors import devo  # Assuming this matches your environment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import DistilBertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = 'distilbert-base-uncased' 
TOP_K_NEIGHBORS = 10      
BLOCKING_THRESHOLD = 0.85 

output_folder = 'blocking'
if not os.path.exists(output_folder): os.makedirs(output_folder)
print(f"Loading Tokenizer from Model: {MODEL_NAME}...")
model = DistilBertTokenizer.from_pretrained(MODEL_NAME)

today = datetime.today().strftime('%Y-%m-%d')

# ==========================================
#  TF-IDF Matrix Tokenization + KNN Search
# ==========================================
def run_matrix_blocking(group_df, top_k=10, threshold=0.85):
    if len(group_df) < 2: 
        return pd.DataFrame()

    # Build the Matrix (Tokenization)
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        tokenizer=model.tokenize,  
        token_pattern=None,     
        min_df=1
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(group_df['blocking_text'])
    except ValueError:
        return pd.DataFrame()

    # Nearest Neighbors - top k matches
    nbrs = NearestNeighbors(n_neighbors=min(top_k + 1, len(group_df)), n_jobs=-1, metric='cosine')
    nbrs.fit(tfidf_matrix)
    distances, indices = nbrs.kneighbors(tfidf_matrix)

    results = []
    
    # Iterate through the results matrix -->  indexes shape: (n_rows, top_k)
    for i in range(indices.shape[0]):
        lid_val = group_df.iloc[i]['entty_riad_cd']
        
        for j, neighbor_idx in enumerate(indices[i]):
            if i == neighbor_idx: continue  # Skip self-match
            
            # Cosine similarity and filter by threshold
            sim_score = 1 - distances[i][j]
            if sim_score < threshold: continue

            rid_val = group_df.iloc[neighbor_idx]['entty_riad_cd']
            
            # Avoid simetric duplicates (A-B y B-A)
            if str(lid_val) >= str(rid_val): continue
            
            results.append({
                'entty_riad_cd': lid_val,
                'matched_entty_riad_cd': rid_val,
                'similarity_score': sim_score,
                'cntry': group_df.iloc[i]['cntry']
            })

    return pd.DataFrame(results)

print("\nFetching data from DEVO...")
query = f'''
SELECT entty_riad_cd,
       cntry,
       nm_entty,
       strt,
       cty,
       pstl_cd
FROM crp_riad.riad_entty_d_1 
WHERE vld_frm <= '{today}' 
  AND vld_t >= '{today}'
  AND entty_riad_cd IS NOT NULL
  AND cntry NOT IN ('__', 'AT','BE','BG','CY','CZ','DE','DK', 'EE', 'ES', 'FI','FR', 'GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK')
'''

df_all = devo.read_sql(query).fillna('')

df_all['blocking_text'] = (df_all['nm_entty']).str.lower().str.strip()

all_candidates = []

countries = df_all['cntry'].unique()
start_time = time()

for country in countries:
    country_df = df_all[df_all['cntry'] == country].copy().reset_index(drop=True)
    
    print(f"Processing {country} ({len(country_df)} entities)...", end=" ")
    t0 = time()
    
    country_matches = run_matrix_blocking(country_df, top_k=TOP_K_NEIGHBORS, threshold=BLOCKING_THRESHOLD)
    
    t1 = time()
    print(f"found {len(country_matches)} pairs in {t1-t0:.2f}s")
    
    if not country_matches.empty:
        all_candidates.append(country_matches)

if all_candidates:
    final_df = pd.concat(all_candidates, ignore_index=True)
    print(f"Total Candidate Pairs Found: {len(final_df)}")
    
    filename = f"blocking_results/matrix_blocking_candidates_{today}.csv"
    final_df.to_csv(filename, index=False)
    
    total_time = time() - start_time
    
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Saved to: {filename}")