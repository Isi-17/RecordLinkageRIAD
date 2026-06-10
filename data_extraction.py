import pandas as pd
from datetime import datetime
import os
import datetime
from connectors import devo

today=datetime.datetime.today()
yesterday=today-datetime.timedelta(days=1)

today=today.strftime('%Y-%m-%d')
yesterday=yesterday.strftime('%Y-%m-%d')

# today

## query to retrieve all extra-EU entities with IDs
query = f'''
WITH IDENTIFIERS AS (
  SELECT entty_riad_id, 
   group_concat(distinct entty_cd) ID_CONCAT
  FROM crp_riad.riad_entty_cd_d_1 
  WHERE vld_frm <=  '{today}'
    AND vld_t >= '{today}'
    AND typ_entty_cd NOT LIKE '%NCB%'
    AND typ_entty_cd <> 'RIAD'
  GROUP BY entty_riad_id
),

OTHER_IDENTIFIERS AS (
  SELECT entty_riad_id, 
   group_concat(distinct entty_cd_othr) OTHER_ID_CONCAT 
  FROM crp_riad.riad_entty_cd_othr_d_1 
  WHERE vld_frm <= '{today}'
    AND vld_t >= '{today}'
     AND lower(entty_cd_othr) NOT LIKE '%notap%'
  GROUP BY entty_riad_id
)


SELECT ENT.entty_riad_cd,
ENT.cntry,
ENT.dt_brth,
ENT.dt_cls,
ENT.nm_entty,
ENT.strt,
ENT.pstl_cd,
ENT.cty,
IDENTIFIERS.ID_CONCAT identifiers,
OTHER_IDENTIFIERS.OTHER_ID_CONCAT other_identifiers
FROM crp_riad.riad_entty_d_1 ENT
  LEFT JOIN IDENTIFIERS
    ON ENT.entty_riad_id = IDENTIFIERS.entty_riad_id
  LEFT JOIN OTHER_IDENTIFIERS
    ON ENT.entty_riad_id = OTHER_IDENTIFIERS.entty_riad_id
WHERE ENT.vld_frm <= '{today}'
  AND ENT.vld_t >= '{today}'
  AND cntry NOT IN ('__', 'AT','BE','BG','CY','CZ','DE','DK', 'EE', 'ES', 'FI','FR', 'GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK')
  AND entty_riad_cd IS NOT NULL
ORDER BY ENT.entty_riad_cd
'''
df_devo_entities =  devo.read_sql(query)

df_devo_entities = df_devo_entities.fillna('')
# df_devo_entities

## query to retrieve all matches based on IDs. The same match
query_id_blocking = f'''
WITH IDENTIFIERS AS (
  SELECT ENT.cntry,
    ID.typ_entty_cd,
    ID.entty_riad_id, 
    ID.entty_cd
  FROM crp_riad.riad_entty_cd_d_1 ID
    JOIN crp_riad.riad_entty_d_1 ENT
      ON ENT.entty_riad_id = ID.entty_riad_id
      AND ENT.vld_frm = ID.vld_frm
  WHERE ID.vld_frm <=  '{today}'
    AND ID.vld_t >= '{today}'
    AND ID.typ_entty_cd NOT LIKE '%NCB%'
    AND ID.typ_entty_cd <> 'RIAD'
),

OTHER_IDENTIFIERS AS (
  SELECT ENT.cntry,
    OTH.typ_entty_cd_othr,
    OTH.entty_riad_id, 
    OTH.entty_cd_othr
  FROM crp_riad.riad_entty_cd_othr_d_1 OTH
  JOIN crp_riad.riad_entty_d_1 ENT
      ON ENT.entty_riad_id = OTH.entty_riad_id
      AND ENT.vld_frm = OTH.vld_frm
  WHERE OTH.vld_frm <= '{today}'
    AND OTH.vld_t >= '{today}'
     AND lower(entty_cd_othr) NOT LIKE '%notap%'
)

--ID vs ID match
SELECT *
FROM (
SELECT DISTINCT ENT.entty_riad_cd,
ENT.cntry,
ENT.dt_brth,
ENT.dt_cls,
ENT.nm_entty,
ENT.strt,
ENT.pstl_cd,
ENT.cty,
ENT2.entty_riad_cd MATCHED_entty_riad_cd,
ENT2.cntry AS MATCHED_cntry,
ENT2.dt_brth MATCHED_dt_brth,
ENT2.dt_cls MATCHED_dt_cls,
ENT2.nm_entty MATCHED_nm_entty,
ENT2.strt MATCHED_strt,
ENT2.pstl_cd MATCHED_pstl_cd,
ENT2.cty MATCHED_cty
FROM crp_riad.riad_entty_d_1 ENT
  LEFT JOIN IDENTIFIERS ID1
    ON ENT.entty_riad_id = ID1.entty_riad_id
  JOIN IDENTIFIERS ID2
    ON ID2.entty_cd = ID1.entty_cd
    AND ID2.cntry = ID1.cntry
    AND ID2.entty_riad_id <> ID1.entty_riad_id
  JOIN crp_riad.riad_entty_d_1 ENT2
    ON ENT2.entty_riad_id = ID2.entty_riad_id
    AND ENT2.vld_frm <= '{today}'
    AND ENT2.vld_t >= '{today}'
WHERE ENT.vld_frm <= '{today}'
  AND ENT.vld_t >= '{today}'
  AND ENT.cntry NOT IN ('__', 'AT','BE','BG','CY','CZ','DE','DK', 'EE', 'ES', 'FI','FR', 'GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK')
  AND ENT.entty_riad_cd IS NOT NULL


UNION

--ID vs OTH match
SELECT DISTINCT ENT.entty_riad_cd,
ENT.cntry,
ENT.dt_brth,
ENT.dt_cls,
ENT.nm_entty,
ENT.strt,
ENT.pstl_cd,
ENT.cty,
ENT2.entty_riad_cd MATCHED_entty_riad_cd,
ENT2.cntry AS MATCHED_cntry,
ENT2.dt_brth MATCHED_dt_brth,
ENT2.dt_cls MATCHED_dt_cls,
ENT2.nm_entty MATCHED_nm_entty,
ENT2.strt MATCHED_strt,
ENT2.pstl_cd MATCHED_pstl_cd,
ENT2.cty MATCHED_cty
FROM crp_riad.riad_entty_d_1 ENT
  LEFT JOIN IDENTIFIERS ID1
    ON ENT.entty_riad_id = ID1.entty_riad_id
  JOIN OTHER_IDENTIFIERS ID2
    ON ID2.entty_cd_othr = ID1.entty_cd
    AND ID2.cntry = ID1.cntry
    AND ID2.entty_riad_id <> ID1.entty_riad_id
  JOIN crp_riad.riad_entty_d_1 ENT2
    ON ENT2.entty_riad_id = ID2.entty_riad_id
    AND ENT2.vld_frm <= '{today}'
    AND ENT2.vld_t >= '{today}'
WHERE ENT.vld_frm <= '{today}'
  AND ENT.vld_t >= '{today}'
  AND ENT.cntry NOT IN ('__', 'AT','BE','BG','CY','CZ','DE','DK', 'EE', 'ES', 'FI','FR', 'GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK')
  AND ENT.entty_riad_cd IS NOT NULL

UNION

--OTH vs OTH match
SELECT DISTINCT ENT.entty_riad_cd,
ENT.cntry,
ENT.dt_brth,
ENT.dt_cls,
ENT.nm_entty,
ENT.strt,
ENT.pstl_cd,
ENT.cty,
ENT2.entty_riad_cd MATCHED_entty_riad_cd,
ENT2.cntry AS MATCHED_cntry,
ENT2.dt_brth MATCHED_dt_brth,
ENT2.dt_cls MATCHED_dt_cls,
ENT2.nm_entty MATCHED_nm_entty,
ENT2.strt MATCHED_strt,
ENT2.pstl_cd MATCHED_pstl_cd,
ENT2.cty MATCHED_cty
FROM crp_riad.riad_entty_d_1 ENT
  LEFT JOIN OTHER_IDENTIFIERS ID1
    ON ENT.entty_riad_id = ID1.entty_riad_id
  JOIN OTHER_IDENTIFIERS ID2
    ON ID2.entty_cd_othr = ID1.entty_cd_othr
    AND ID2.cntry = ID1.cntry
    AND ID2.entty_riad_id <> ID1.entty_riad_id
  JOIN crp_riad.riad_entty_d_1 ENT2
    ON ENT2.entty_riad_id = ID2.entty_riad_id
    AND ENT2.vld_frm <= '{today}'
    AND ENT2.vld_t >= '{today}'
WHERE ENT.vld_frm <= '{today}'
  AND ENT.vld_t >= '{today}'
  AND ENT.cntry NOT IN ('__', 'AT','BE','BG','CY','CZ','DE','DK', 'EE', 'ES', 'FI','FR', 'GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK')
  AND ENT.entty_riad_cd IS NOT NULL
) MY_SELECTION
ORDER BY entty_riad_cd
'''
df_id_blocking =  devo.read_sql(query_id_blocking)

df_id_blocking = df_id_blocking.fillna('')
# df_id_blocking



df_merged_all = df_id_blocking.merge(df_devo_entities, 
                                     left_on= ['entty_riad_cd', 'cntry', 'dt_brth', 'dt_cls', 'nm_entty', 'strt', 'pstl_cd', 'cty'],
                                     right_on= ['entty_riad_cd', 'cntry', 'dt_brth', 'dt_cls', 'nm_entty', 'strt', 'pstl_cd', 'cty']
                                    )
df_merged_all = df_merged_all.merge(df_devo_entities, 
                                     left_on= ['matched_entty_riad_cd', 'matched_cntry', 'matched_dt_brth', 'matched_dt_cls', 'matched_nm_entty', 'matched_strt', 'matched_pstl_cd', 'matched_cty'],
                                     right_on= ['entty_riad_cd', 'cntry', 'dt_brth', 'dt_cls', 'nm_entty', 'strt', 'pstl_cd', 'cty']
                                    )
df_id_blocking_new = df_merged_all[['entty_riad_cd_x', 'cntry_x', 'dt_brth_x', 'dt_cls_x', 'nm_entty_x', 'strt_x', 'pstl_cd_x', 'cty_x',
                                   'identifiers_x', 'other_identifiers_x', 
                                  'matched_entty_riad_cd', 'matched_cntry', 'matched_dt_brth', 'matched_dt_cls', 'matched_nm_entty', 'matched_strt', 'matched_pstl_cd', 'matched_cty',
                                   'identifiers_y', 'other_identifiers_y'
                                  ]]
df_id_blocking_new.columns = ['entty_riad_cd', 'cntry', 'dt_brth', 'dt_cls', 'nm_entty', 'strt', 'pstl_cd', 'cty', 'identifiers', 'other_identifiers',
'matched_entty_riad_cd', 'matched_cntry', 'matched_dt_brth', 'matched_dt_cls', 'matched_nm_entty', 'matched_strt', 'matched_pstl_cd', 'matched_cty','matched_identifiers', 'matched_other_identifiers']


output_folder = 'DEVO_blocking_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the ID Blocking results
id_blocking_filename = f"DEVO_blocking_results/id_blocking_output_{today}.csv"
df_id_blocking_new.to_csv(id_blocking_filename, index=False, encoding='utf-8-sig')

# df_id_blocking_new





from time import time

start_time = time()

countries = df_devo_entities.cntry.unique()

list_name_blocking_dfs = []

for country in countries:
    mid_time = time()
    print(country)


    country_results = []
    
    query_riad_codes = f"""
    SELECT entty_riad_cd
    FROM crp_riad.riad_entty_d_1 ENT
    WHERE ENT.vld_frm <= '{today}'
      AND ENT.vld_t >= '{today}'
      AND cntry = '{country}'
      AND entty_riad_cd IS NOT NULL
    """

    riad_codes = devo.read_sql(query_riad_codes)
    
    tot_entities = len(riad_codes)
    index = 0
    
    if tot_entities < 20000:
        query_name_blocking = f'''
            WITH IDENTIFIERS AS (
              SELECT entty_riad_id, 
               group_concat(distinct entty_cd) ID_CONCAT
              FROM crp_riad.riad_entty_cd_d_1 
              WHERE vld_frm <=  '{today}'
                AND vld_t >= '{today}'
                AND typ_entty_cd NOT LIKE '%NCB%'
                AND typ_entty_cd <> 'RIAD'
              GROUP BY entty_riad_id
            ),

            OTHER_IDENTIFIERS AS (
              SELECT entty_riad_id, 
               group_concat(distinct entty_cd_othr) OTHER_ID_CONCAT 
              FROM crp_riad.riad_entty_cd_othr_d_1 
              WHERE vld_frm <= '{today}'
                AND vld_t >= '{today}'
                 AND lower(entty_cd_othr) NOT LIKE '%notap%'
              GROUP BY entty_riad_id
            ),

            ENTITIES as (
            SELECT ENT.entty_riad_cd,
            ENT.cntry,
            ENT.dt_brth,
            ENT.dt_cls,
            ENT.nm_entty,
            ENT.strt,
            ENT.pstl_cd,
            ENT.cty,
            IDENTIFIERS.ID_CONCAT identifiers,
            OTHER_IDENTIFIERS.OTHER_ID_CONCAT other_identifiers
            FROM crp_riad.riad_entty_d_1 ENT
              LEFT JOIN IDENTIFIERS
                ON ENT.entty_riad_id = IDENTIFIERS.entty_riad_id
              LEFT JOIN OTHER_IDENTIFIERS
                ON ENT.entty_riad_id = OTHER_IDENTIFIERS.entty_riad_id
            WHERE ENT.vld_frm <= '{today}'
              AND ENT.vld_t >= '{today}'
              AND ENT.cntry = '{country}'
              AND ENT.entty_riad_cd IS NOT NULL
            )


            SELECT *
            FROM (
              SELECT ORIGINAL.*,
              PAIRED.entty_riad_cd matched_entty_riad_cd,
              PAIRED.cntry matched_cntry,
              PAIRED.dt_brth matched_dt_brth,
              PAIRED.dt_cls matched_dt_cls,
              PAIRED.nm_entty matched_nm_entty,
              PAIRED.strt matched_strt,
              PAIRED.pstl_cd matched_pstl_cd,
              PAIRED.cty matched_cty,
              PAIRED.identifiers  matched_identifiers,
              PAIRED.other_identifiers matched_other_identifiers,
              JARO_WINKLER_SIMILARITY(LEFT(ORIGINAL.nm_entty, 255), LEFT(PAIRED.nm_entty,255)) SIMILARITY_SCORE,
              ROW_NUMBER() OVER (PARTITION BY ORIGINAL.entty_riad_cd ORDER BY JARO_WINKLER_SIMILARITY(LEFT(ORIGINAL.nm_entty, 255), LEFT(PAIRED.nm_entty,255)) DESC) PAIR_COUNT
              FROM ENTITIES ORIGINAL
                JOIN ENTITIES PAIRED
                  ON ORIGINAL.cntry = PAIRED.cntry
                  AND ORIGINAL.entty_riad_cd < PAIRED.entty_riad_cd
            ) my_selection
            WHERE SIMILARITY_SCORE >= 0.85
              AND PAIR_COUNT <= 10
            ORDER BY entty_riad_cd
            '''
        
        df_name_blocking =  devo.read_sql(query_name_blocking)
        list_name_blocking_dfs.append(df_name_blocking)
        total_time = time() - mid_time
        print(f"Total execution time for this block: {total_time:.2f} seconds.")

        country_results.append(df_name_blocking)

    else:
        while index < tot_entities:

            riad_codes_filter = '" , "'.join(map(str, riad_codes['entty_riad_cd'][range(index, min(tot_entities,index+2000))])) 
            riad_codes_filter = ' ("' + riad_codes_filter + '")'

            query_name_blocking = f'''
                WITH IDENTIFIERS AS (
                  SELECT entty_riad_id, 
                   group_concat(distinct entty_cd) ID_CONCAT
                  FROM crp_riad.riad_entty_cd_d_1 
                  WHERE vld_frm <=  '{today}'
                    AND vld_t >= '{today}'
                    AND typ_entty_cd NOT LIKE '%NCB%'
                    AND typ_entty_cd <> 'RIAD'
                  GROUP BY entty_riad_id
                ),

                OTHER_IDENTIFIERS AS (
                  SELECT entty_riad_id, 
                   group_concat(distinct entty_cd_othr) OTHER_ID_CONCAT 
                  FROM crp_riad.riad_entty_cd_othr_d_1 
                  WHERE vld_frm <= '{today}'
                    AND vld_t >= '{today}'
                     AND lower(entty_cd_othr) NOT LIKE '%notap%'
                  GROUP BY entty_riad_id
                ),

                ENTITIES as (
                SELECT ENT.entty_riad_cd,
                ENT.cntry,
                ENT.dt_brth,
                ENT.dt_cls,
                ENT.nm_entty,
                ENT.strt,
                ENT.pstl_cd,
                ENT.cty,
                IDENTIFIERS.ID_CONCAT identifiers,
                OTHER_IDENTIFIERS.OTHER_ID_CONCAT other_identifiers
                FROM crp_riad.riad_entty_d_1 ENT
                  LEFT JOIN IDENTIFIERS
                    ON ENT.entty_riad_id = IDENTIFIERS.entty_riad_id
                  LEFT JOIN OTHER_IDENTIFIERS
                    ON ENT.entty_riad_id = OTHER_IDENTIFIERS.entty_riad_id
                WHERE ENT.vld_frm <= '{today}'
                  AND ENT.vld_t >= '{today}'
                  AND ENT.cntry = '{country}'
                  AND ENT.entty_riad_cd IS NOT NULL
                )


                SELECT *
                FROM (
                  SELECT ORIGINAL.*,
                  PAIRED.entty_riad_cd matched_entty_riad_cd,
                  PAIRED.cntry matched_cntry,
                  PAIRED.dt_brth matched_dt_brth,
                  PAIRED.dt_cls matched_dt_cls,
                  PAIRED.nm_entty matched_nm_entty,
                  PAIRED.strt matched_strt,
                  PAIRED.pstl_cd matched_pstl_cd,
                  PAIRED.cty matched_cty,
                  PAIRED.identifiers  matched_identifiers,
                  PAIRED.other_identifiers matched_other_identifiers,
                  JARO_WINKLER_SIMILARITY(LEFT(ORIGINAL.nm_entty, 255), LEFT(PAIRED.nm_entty,255)) SIMILARITY_SCORE,
                  ROW_NUMBER() OVER (PARTITION BY ORIGINAL.entty_riad_cd ORDER BY JARO_WINKLER_SIMILARITY(LEFT(ORIGINAL.nm_entty, 255), LEFT(PAIRED.nm_entty,255)) DESC) PAIR_COUNT
                  FROM ENTITIES ORIGINAL
                    JOIN ENTITIES PAIRED
                      ON ORIGINAL.cntry = PAIRED.cntry
                      AND ORIGINAL.entty_riad_cd < PAIRED.entty_riad_cd
                  WHERE ORIGINAL.entty_riad_cd IN {riad_codes_filter}
                ) my_selection
                WHERE SIMILARITY_SCORE >= 0.85
                  AND PAIR_COUNT <= 10
                ORDER BY entty_riad_cd
                '''

            df_name_blocking =  devo.read_sql(query_name_blocking)
            list_name_blocking_dfs.append(df_name_blocking)
            total_time = time() - mid_time
            print(f"Total execution time for this block: {total_time:.2f} seconds.")

            country_results.append(df_name_blocking)

            index = index + 2000

    if country_results:
        df_country_final = pd.concat(country_results).fillna('')
        
        file_name = f"name_blocking_{country}_{today}.csv"
        file_path = os.path.join(output_folder, file_name)
        
        df_country_final.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Saved {len(df_country_final)} matches for {country} to {file_path}")


end_time = time()
total_time = end_time - start_time
print(f"Finished in {total_time:.2f} seconds")
