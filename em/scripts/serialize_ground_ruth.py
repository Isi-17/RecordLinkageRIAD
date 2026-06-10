import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

input_jsonl = 'ground_truth.jsonl' 
dataset_name = 'riad_entities'
output_dir = f'data/raw/ditto_files/{dataset_name}'

os.makedirs(output_dir, exist_ok=True)

COLUMNS_MAPPING = [
    ('NM_ENTTY', 'name'),
    ('STRT', 'street'),
    ('PSTL_CD', 'postal_code'),
    ('CTY', 'city'),
    # ('RIAD_CD', 'riad_id'),
    ('ID', 'ids_concat')
]

def serialize_entity(row, side):
    """
    Serialize each entity taking multiple IDs in the same field.
    """
    suffix = f"_{side}"
    tokens = []
    
    for original_base, target_name in COLUMNS_MAPPING:
        original_key = f"{original_base}{suffix}"
        
        # Default
        final_val_str = ""
        
        # ID field could be a dictionary
        if original_base == 'ID':
            id_data = row.get(original_key)
            if isinstance(id_data, dict):
                # We gather all attributes
                valid_ids = []
                for k, v in id_data.items():
                    if v and str(v).lower() != 'nan':
                        clean_id = str(v).replace('\t', '').strip()
                        valid_ids.append(clean_id)
                
                # ID joinning
                if valid_ids:
                    final_val_str = " ".join(valid_ids)
            
            elif id_data and str(id_data).lower() != 'nan':
                 final_val_str = str(id_data).strip()
                 
        else:
            raw_val = row.get(original_key)
            if raw_val and str(raw_val).lower() != 'nan':
                final_val_str = str(raw_val).replace('\t', ' ').replace('\n', ' ').strip()
        
        tokens.append(f"COL {target_name} VAL {final_val_str}")

    return " ".join(tokens)

data = []
print("Processing all IDs...")

try:
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            
            left_str = serialize_entity(record, 1)
            right_str = serialize_entity(record, 2)
            
            label = 1 if record['ASSESSMENT'] == 'y' else 0
            
            data.append([left_str, right_str, label])
except FileNotFoundError:
    print(f"Error: No se encuentra el fichero {input_jsonl}")
    exit()

df = pd.DataFrame(data, columns=['left', 'right', 'label'])

print(f"Processing {len(df)} candidate pairs...")
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

def save_ditto_file(dataframe, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            f.write(f"{row['left']}\t{row['right']}\t{row['label']}\n")

save_ditto_file(train_df, os.path.join(output_dir, 'train.txt'))
save_ditto_file(val_df, os.path.join(output_dir, 'valid.txt'))
save_ditto_file(test_df, os.path.join(output_dir, 'test.txt'))