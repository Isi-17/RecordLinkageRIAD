import os
import pandas as pd
import json

def parse_id_list(id_list_strings):
    """
    Parses a list of strings. Handles multiple IDs on the same line.
    Example input: ['us_dsfn_cd=123 lei=456']
    Output: {'us_dsfn_cd': '123', 'lei': '456'}
    """
    id_dict = {}
    for line in id_list_strings:
        tokens = line.split()
        for token in tokens:
            token = token.strip()
            if not token: continue

            if '=' in token:
                key, val = token.split('=', 1)
                id_dict[key.strip()] = val.strip()
            else:
                id_dict[f"raw_{token}"] = token

    return id_dict

def get_split_values(raw_value):
    """
    Splits a string into (Value 1, Value 2).
    Case 1: Split by '----------//----------'
    Case 2: Split by Newline '\n'
    """
    val_str = str(raw_value).strip()
    delimiter = '----------//----------'

    if not val_str or val_str.lower() == 'nan':
        return '', ''

    if delimiter in val_str:
        parts = val_str.split(delimiter)
        v1 = parts[0].strip() if len(parts) > 0 else ''
        v2 = parts[1].strip() if len(parts) > 1 else ''
        return v1, v2
    
    elif '\n' in val_str:
        parts = val_str.split('\n', 1)
        v1 = parts[0].strip()
        v2 = parts[1].strip() if len(parts) > 1 else ''
        return v1, v2

    # 3. No separator found -> Everything belongs to Entity 1
    else:
        return val_str, ''

with open('data/ground_truth/ground_truth_AssessmentFiles.jsonl', 'w', encoding='utf-8') as f_out:
    folder_path = 'data/ground_truth/attic - resolved cases'
    delimiter = '----------//----------'
    seen_pairs = set()
    unique_entities = set()
    no_valid_assessment_count = set()
    used_files = []
    not_processed_files = []
    not_used_files = []
    assessment_count = {'y': 0, 'n': 0}

    for filename in os.listdir(folder_path):
        if not (filename.endswith('.xlsx') or filename.endswith('.xls')):
            continue
        try:
            used_files.append(filename)
            xls = pd.ExcelFile(os.path.join(folder_path, filename))
            relevant_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith(('ex_post', 'Tmp-Tmp_', 'Tmp-Of', 'Tmp-vs-Of', 'Tmp-vs-Tmp'))]
            if not relevant_sheets:
                not_used_files.append(filename)
                # print(f"No relevant sheets in file {filename}, skipping.")
                continue
            for sheet_name in relevant_sheets:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                unique_assessments = set(df['Assessment'].astype(str).str.lower().unique())
                if not unique_assessments.intersection({'y', 'n', 'yes', 'no'}):
                    no_valid_assessment_count.add(filename)
                    # print(f"Skipping {sheet_name} form file {filename} (No valid assessments found)")
                    continue

                for index, row in df.iterrows():
                    if sheet_name.startswith('ex_post'):
                        score = 1
                    else:
                        score = str(row.get('evidence', ''))
                    
                    assessment = str(row.get('Assessment', '')).strip().lower()
                    if assessment == 'yes':
                        assessment = 'y'
                    elif assessment == 'no':
                        assessment = 'n'
                    if assessment not in ('y', 'n'):
                        # print(f"Invalid assessment value '{assessment}' in file {filename}, sheet {sheet_name}, row {index}, skipping.")
                        continue
                    assessment_count[assessment] += 1

                    r1 = str(row.get('entty_riad_cd_1', '')).strip()
                    r2 = str(row.get('entty_riad_cd_2', '')).strip()

                    # Check if already seen
                    pair_signature = tuple(sorted([r1, r2]))
                    if pair_signature in seen_pairs:
                        # print(f"Duplicate pair found: {pair_signature}, skipping.")
                        continue
                        
                    seen_pairs.add(pair_signature)
                    unique_entities.add(r1)
                    unique_entities.add(r2)

                    nm_1, nm_2 = get_split_values(row.get('nm_entty_cmp', ''))
                    strt_1, strt_2 = get_split_values(row.get('strt_cmp', ''))
                    pstl_1, pstl_2 = get_split_values(row.get('pstl_cd_cmp', ''))
                    cty_1, cty_2 = get_split_values(row.get('cty_cmp', ''))

                    # 2. IDs
                    id_raw = str(row.get('id_cmp', ''))
                    if delimiter in id_raw:
                        raw_parts = id_raw.split(delimiter)
                        # entity 1
                        block_1 = raw_parts[0].strip()
                        list_1 = [block_1] if block_1 else []
                        
                        # entity 2
                        block_2 = raw_parts[1].strip() if len(raw_parts) > 1 else ''
                        list_2 = [block_2] if block_2 else []
                    
                    elif '\n' in id_raw:
                            parts = id_raw.split('\n', 1)
                            list_1 = [parts[0].strip()] if parts[0].strip() else []
                            list_2 = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
                    else:
                        # No separator found -> everything goes to ID_1
                        list_1 = [id_raw.strip()] if id_raw.strip() else []
                        list_2 = []

                    # Parse lists (now handles spaces correctly)
                    dict_id_1 = parse_id_list(list_1)
                    dict_id_2 = parse_id_list(list_2)
                    
                    result_json = {
                        "RIAD_CD_1": r1,
                        "RIAD_CD_2": r2,
                        "NM_ENTTY_1": nm_1,
                        "NM_ENTTY_2": nm_2,
                        "ID_1": dict_id_1,
                        "ID_2": dict_id_2,
                        "STRT_1": strt_1,
                        "STRT_2": strt_2,
                        "PSTL_CD_1": pstl_1,
                        "PSTL_CD_2": pstl_2,
                        "CTY_1": cty_1,
                        "CTY_2": cty_2,
                        "SCORE": score,
                        "ASSESSMENT": assessment
                    }

                    f_out.write(json.dumps(result_json, ensure_ascii=False) + '\n')

            used_files.append(filename)

        except Exception as e:
            not_processed_files.append(filename)
            print(f"Error processing file {filename}: {e}")

print("-" * 30)
print(f"Files Processed:                                  {len(used_files)}")
print(f"Files Not Processed:                              {len(not_processed_files)}")
print(f"Files processed but not used (wrong sheet names): {len(not_used_files)}")
print(f"Sheets Skipped (No Valid Assessment):             {len(no_valid_assessment_count)}")
print(f"Total Unique Cases (Pairs):                       {len(seen_pairs)}")
print(f"Total Distinct Entities:                          {len(unique_entities)}")
print(f"Assessment Counts:                                {assessment_count}")
print(f"Ratio y:                                          {assessment_count['y'] / (assessment_count['y'] + assessment_count['n']) * 100:.2f}%")
print(f"Ratio n:                                          {assessment_count['n'] / (assessment_count['y'] + assessment_count['n']) * 100:.2f}%")
print("-" * 30)