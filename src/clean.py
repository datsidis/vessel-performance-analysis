from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "CPS_Poseidon.parquet"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(RAW)
n0 = len(df)

# Rule 1: both propeller sensors must be present
df = df.dropna(subset=["Propeller_Port_ShaftPower", "Propeller_Starboard_ShaftPower"])
print(f"After both-propellers rule: {len(df)} rows ({n0 - len(df)} removed)")

# Rule 2: rebuild total power ourselves instead of trusting the 'Total' column
df["power_mw"] = (df["Propeller_Port_ShaftPower"] + df["Propeller_Starboard_ShaftPower"]) / 1e6
df["stw_kn"] = df["Ship_SpeedThroughWater"] * 1.944

# Rule 3: steady sailing only (no port, no manoeuvring)
n1 = len(df)
df = df[(df["stw_kn"] > 8) & (df["power_mw"] > 0.5)]
print(f"After sailing-only rule: {len(df)} rows ({n1 - len(df)} removed)")

df.to_parquet(OUT / "poseidon_clean_v1.parquet")
print(f"Kept {len(df) / n0:.1%} of the original data")

# Rule 4: remove frozen power readings (same exact value many rows in a row)
n2 = len(df)
same_as_prev = df["Propeller_Port_ShaftPower"].diff().eq(0)
run_id = (~same_as_prev).cumsum()
run_len = df.groupby(run_id)["Propeller_Port_ShaftPower"].transform("size")
df = df[run_len < 6]   # 6+ identical readings in a row = frozen sensor
print(f"After frozen-sensor rule: {len(df)} rows ({n2 - len(df)} removed)")
for col in ["Propeller_Port_ShaftPower", "Ship_SpeedThroughWater"]:
    same = df[col].diff().eq(0)
    run_len = df.groupby((~same).cumsum())[col].transform("size")
    df = df[run_len < 3]

# Diagnostic: inspect the 13.8 MW streak
streak = df[(df["power_mw"].between(13.6, 14.0)) & (df["stw_kn"].between(14, 17))]
print(len(streak), "rows in the streak")
print(streak[["index", "stw_kn", "power_mw",
              "Propeller_Port_ShaftPower",
              "Propeller_Starboard_ShaftPower"]].head(20).to_string())

# Was the speed loss caused by weather?
cols = ["index", "stw_kn", "power_mw", "Ship_SpeedOverGround",
        "Weather_WindSpeed10M", "Weather_WaveHeight",
        "Weather_OceanCurrentVelocity", "Ship_Heading", "Weather_WindDirection10M"]
print(df.loc[df["index"].between(28120, 28145), cols].to_string())