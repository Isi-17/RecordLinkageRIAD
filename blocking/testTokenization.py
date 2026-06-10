import pandas as pd
import numpy as np
from datetime import datetime
from connectors import devo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import time

TOP_K_NEIGHBORS = 10    
BLOCKING_THRESHOLD = 0.8

output_folder = 'blocking'
if not os.path.exists(output_folder): os.makedirs(output_folder)
    
today = datetime.today().strftime('%Y-%m-%d')

# ============================
#     n-gram TF-IDF matrix
# ============================
def run_vector_blocking(group_df, top_k=10, threshold=0.8):
    if len(group_df) < 2: return pd.DataFrame()

    # Vectorization TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(group_df['blocking_text'])
    except ValueError:
        return pd.DataFrame()

    # Nearest Neighbors - top k matches
    nbrs = NearestNeighbors(n_neighbors=min(top_k + 1, len(group_df)), n_jobs=-1, metric='cosine')
    nbrs.fit(tfidf_matrix)
    distances, indexes = nbrs.kneighbors(tfidf_matrix)

    results = []
    
    # Iterate through the results matrix -->  indexes shape: (n_rows, top_k)
    for i in range(indexes.shape[0]):
        for j, neighbor_idx in enumerate(indexes[i]):
            if i == neighbor_idx: continue # Skip self-match
            
            # Cosine similarity and filter by threshold
            score = 1 - distances[i][j]
            if score < threshold: continue

            lid = group_df.iloc[i]['entty_riad_cd']
            rid = group_df.iloc[neighbor_idx]['entty_riad_cd']
            
            # Avoid simetric duplicates (A-B y B-A)
            if lid >= rid: continue

            results.append({
                'entty_riad_cd': lid,
                'matched_entty_riad_cd': rid,
                'cntry': group_df.iloc[i]['cntry'],
                'similarity_score': score
            })

    return pd.DataFrame(results)

print("Fetching data from DEVO...")
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

print(f"Starting blocking for {len(countries)} countries...")
total_start = time.time()

for country in countries:
    country_df = df_all[df_all['cntry'] == country].copy().reset_index(drop=True)
    
    print(f"Processing {country} ({len(country_df)} entities)...")
    t0 = time.time()
    
    matches = run_vector_blocking(country_df, top_k=TOP_K_NEIGHBORS, threshold=BLOCKING_THRESHOLD)
    t1 = time.time()
    elapsed = t1 - t0
    
    if not matches.empty:
        print(f"   Saved {len(matches)} matches.")
        print(f"   > Found {len(matches)} candidates in {elapsed:.2f}s ({len(matches)/elapsed:.0f} pairs/sec)")
        
        all_results.append(matches)

if all_results:
    final_df = pd.concat(all_results)
    print(f"Total Candidate Pairs Found: {len(final_df)} in {time.time() - total_start:.1f}s")
    
    path = os.path.join(output_folder, f"vector_blocking_results_{today}.csv")
    final_df.to_csv(path, index=False)
    print(f"Saved to {path}")