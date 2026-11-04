from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "processed" / "poseidon_features_v1.parquet"
OUT = ROOT / "data" / "processed" / "poseidon_baseline_v1.parquet"
FIGS = ROOT / "outputs" / "figures"

df = pd.read_parquet(FEAT)

# Time proxy: rows assumed 5 minutes apart (check README, change if needed)
SAMPLE_MIN = 5
df["day"] = df["index"] * SAMPLE_MIN / 1440

# 1. Calm-weather filter (ISO 19030 idea: compare like with like)
calm_mask = (
    (df["head_wind_ms"].abs() < 5)
    & (df["Weather_WaveHeight"] < 1.5)
    & (df["stw_kn"] >= 10)
)
calm = df[calm_mask]
print(f"Calm-weather rows: {len(calm)} of {len(df)} ({len(calm) / len(df):.1%})")

# 2. Fit P = a * V^n  (linear fit in log space)
n, log_a = np.polyfit(np.log(calm["stw_kn"]), np.log(calm["power_mw"]), 1)
a = np.exp(log_a)
pred = a * calm["stw_kn"] ** n
ss_res = ((calm["power_mw"] - pred) ** 2).sum()
ss_tot = ((calm["power_mw"] - calm["power_mw"].mean()) ** 2).sum()
print(f"Baseline: P = {a:.5f} * V^{n:.2f}   (R² = {1 - ss_res / ss_tot:.3f})")

# 3. Performance indicator for every row
df["power_expected_mw"] = a * df["stw_kn"] ** n
df["excess_power_pct"] = (df["power_mw"] / df["power_expected_mw"] - 1) * 100
df.to_parquet(OUT)

# Plot 1: baseline curve over calm points
v = np.linspace(calm["stw_kn"].min(), calm["stw_kn"].max(), 200)
plt.figure(figsize=(8, 5))
plt.scatter(calm["stw_kn"], calm["power_mw"], s=1, alpha=0.3, label="calm-weather data")
plt.plot(v, a * v ** n, color="black", lw=2, label=f"baseline P = {a:.4f}·V^{n:.2f}")
plt.xlabel("Speed through water (knots)")
plt.ylabel("Propeller shaft power (MW)")
plt.title("Clean-weather speed-power baseline")
plt.legend(markerscale=10)
plt.tight_layout()
plt.savefig(FIGS / "baseline_fit.png", dpi=150)
plt.show()

# Plot 2: excess power over time (calm rows only, so weather is not the cause)
trend = df.loc[calm_mask].set_index("day")["excess_power_pct"].sort_index()
rolling = trend.rolling(500, center=True, min_periods=100).median()
slope_per_month = np.polyfit(trend.index, trend.values, 1)[0] * 30
print(f"Trend: {slope_per_month:+.2f} % excess power per 30 days")

plt.figure(figsize=(10, 5))
plt.scatter(trend.index, trend.values, s=1, alpha=0.15, label="calm-weather rows")
plt.plot(rolling.index, rolling.values, color="black", lw=2, label="rolling median")
plt.axhline(0, color="grey", ls="--")
plt.ylim(-40, 40)
plt.xlabel(f"Day (assumes {SAMPLE_MIN}-min sampling)")
plt.ylabel("Excess power vs baseline (%)")
plt.title("Performance trend over time")
plt.legend(markerscale=10)
plt.tight_layout()
plt.savefig(FIGS / "excess_power_trend.png", dpi=150)
plt.show()

raw = pd.read_parquet(ROOT / "data" / "raw" / "CPS_Poseidon.parquet")
raw["day"] = raw["index"] * SAMPLE_MIN / 1440
gap = raw[raw["day"].between(30, 100)]
daily = gap.groupby(gap["day"].astype(int)).agg(
    mean_stw_kn=("Ship_SpeedThroughWater", lambda s: (s * 1.944).mean()),
    stbd_missing=("Propeller_Starboard_ShaftPower", lambda s: s.isna().mean()),
    mean_wave=("Weather_WaveHeight", "mean"),
)
print(daily.round(2).to_string())