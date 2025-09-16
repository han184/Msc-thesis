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
# intial input list
tags = [
    "time",
    "fr_eng",
    "te_exh_cyl_out__0",
    "pd_air_ic__0",
    "pr_exh_turb_out__0",
    "te_air_ic_out__0",
    "te_seawater",
    "te_air_comp_in_a__0",
    "te_air_comp_in_b__0",
    "fr_tc__0",
    "pr_baro",
    "te_exh_turb_in__0",
    "te_exh_turb_out__0",
    "pr_exh_rec",
    "pr_air_scav",
    "pr_air_scav_ecs",
    "fr_eng_setpoint",
    "pr_cyl_max__0",
    "se_mip_acco__0",
    "se_mip__0",
    "pr_cyl_comp__0",
    "te_cw_ic_in_common",
    "in_stable",
    "te_air_scav_rec",
    "re_perf_idx_hrn_indicator",
    "in_engine_running_mode",
    "bo_aux_blower_running",
    "re_eng_load",
    ]


# final input list
tags_final = [
    "time",
    "fr_eng",
    "te_cw_ic_in_common",
    "pr_cyl_comp__0",
    "te_seawater",
    "te_exh_cyl_out__0",
    "fr_eng_setpoint",
    "pr_air_scav",
    "pr_cyl_max__0",
    "in_stable",
    "in_engine_running_mode",
    "pd_air_ic__0",
    "te_air_ic_out__0",
    "te_air_scav_rec",
    "pr_air_scav_ecs",
    "pr_exh_rec",
    "pr_baro",
    "se_mip_acco__0",
    "se_mip__0",
    "re_eng_load"
    ]


# engine type with the same fuel injection concept
# S50ME-C9.7-HPSCR (Engine A)
# product_id = "b45bf2c1e70db9d43592ecec27143953"
# product_id = "afa07d8b71a911de438a94f88fe50ac8"
# product_id = "2c0966beabf75c257596d8fa26842123"

# G70ME-C10.5-HPSCR (Engine B)
product_id = "b47fecf6346fbd050d467a164dd82be1"

# S60ME-C10.5-HPSCR (Engine C)
# product_id = "29a955027a26c571c3d1a7f252a62c6e"

cache = Path(f'/tmp/data_{get_tags_hash(tags)}_{product_id}_{start}_{stop}.parquet')
if cache.exists():
    df = pd.read_parquet(cache)
else:
    con = setup_duckdb()
    df = load_engine_data(con, product_id, start, stop, tags).df()
    df.to_parquet(cache)

path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/han/dataframe_engine3_1_forimputation"
df.to_parquet(path, index=False)
print(df.to_string())