# create the dataframe
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
import pandas as pd
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib

pd.set_option('display.max_columns', None)

def load_engine_data(
    con: DuckDB, product_id: str, start: pd.Timestamp, stop: pd.Timestamp, tags: list[str]
) -> Relation:
    return con.sql(f"""
    SELECT {','.join(tags)}
    FROM timeseries
    WHERE
        time BETWEEN '{start}' AND '{stop}'
        AND pid = '{product_id}'
    """)

def get_tags_hash(tags):
    return hashlib.md5(','.join(tags).encode()).hexdigest()

start, stop = pd.Timestamp("2023-05-01"), pd.Timestamp("2024-04-30")
tags = [
    "time",
    'fr_eng',
    'te_exh_cyl_out__0',
    'pd_air_ic__0',
    'pr_exh_turb_out__0',
    'te_air_ic_out__0',
    'te_seawater',
    'te_air_comp_in_a__0',
    'te_air_comp_in_b__0',
    'fr_tc__0',
    'pr_baro',
    'te_exh_turb_in__0',
    'te_exh_turb_out__0',
    'pr_exh_rec',
    'pr_air_scav',
    'pr_air_scav_ecs',
    'fr_eng_setpoint',
    'pr_cyl_max__0',
    'se_mip_acco__0',
    'se_mip__0',
    'pr_cyl_comp__0',
    'te_cw_ic_in_common',
    'in_stable',
    'te_air_scav_rec',
    're_perf_idx_hrn_indicator',
    'in_engine_running_mode',
    'bo_aux_blower_running',
    're_eng_load'
    ]
# product_id = "ea52565f40ed312a2f3a18071998ce0a"
# product_id = "93c3fb4d51768854def112a19b2a6d26"
# product_id = "aca212b21eed42f075310163af8e6884"
# product_id = "22161a5effa4f032f76ec9266b74e240"
# product_id = "3d6810f5-e8a2-463c-8eb6-21a1555b08c8"
# product_id = "ce5f489c39cb54de783d7eadd727aa55"
# engine used for training in june
product_id = "d6a77f306e013a579cb4eb3a7ed3571b"
cache = Path(f'/tmp/data_{get_tags_hash(tags)}_{product_id}_{start}_{stop}.parquet')
if cache.exists():
    df = pd.read_parquet(cache)
else:
    con = setup_duckdb()
    df = load_engine_data(con, product_id, start, stop, tags).df()
    df.to_parquet(cache)

path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/han/dataframe_oldid"
df.to_parquet(path, index=False)
print(df.to_string())

