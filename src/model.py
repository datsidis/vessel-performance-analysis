from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "poseidon_baseline_v1.parquet"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA).sort_values("day")

FEATURES = [
    "stw_kn", "head_wind_ms", "Weather_WindSpeed10M", "rel_wind_deg",
    "Weather_WaveHeight", "rel_wave_deg", "Weather_WavePeriod",
    "Weather_SwellWaveHeight", "current_along_ms", "Environment_SeaFloorDepth",
]
TARGET = "power_mw"
REF_END = 130     # reference period: days 0-130 ("younger" hull)
TEST_END = 160    # accuracy test on the next 30 days (little fouling yet)
OUTLIER_PCT = 20  # daily median excess above this = outlier day

ref = df[df["day"] < REF_END]
test = df[(df["day"] >= REF_END) & (df["day"] < TEST_END)]
print(f"Reference rows: {len(ref)}, test rows: {len(test)}")

# ---------- 1. Physics baseline (fitted first: the hybrid model builds on it) ----------
calm = (ref["head_wind_ms"].abs() < 5) & (ref["Weather_WaveHeight"] < 1.5)
n, log_a = np.polyfit(np.log(ref.loc[calm, "stw_kn"]), np.log(ref.loc[calm, "power_mw"]), 1)
a = np.exp(log_a)
print(f"Physics baseline (reference period): P = {a:.5f} * V^{n:.2f}")


def physics(d):
    """Clean-hull, calm-weather power from the cube-law baseline."""
    return a * d["stw_kn"] ** n


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    )


# ---------- 2. Train two ML models ----------
# Pure ML: predicts power directly (cannot extrapolate beyond training speeds)
model_direct = make_model()
model_direct.fit(ref[FEATURES], ref[TARGET])

# Hybrid: physics handles speed, ML learns only the weather correction factor
ref_ratio = ref[TARGET] / physics(ref)
model_hybrid = make_model()
model_hybrid.fit(ref[FEATURES], ref_ratio)


def predict_hybrid(d):
    return physics(d) * model_hybrid.predict(d[FEATURES])


# ---------- 3. Accuracy comparison ----------
y = test[TARGET]
preds = {
    "Physics P=a*V^n": physics(test),
    "Pure ML": model_direct.predict(test[FEATURES]),
    "Hybrid (physics x ML)": predict_hybrid(test),
}
print(f"\nAccuracy on unseen days {REF_END}-{TEST_END}:")
for name, p in preds.items():
    mae = mean_absolute_error(y, p)
    mape = (np.abs(p - y) / y).mean() * 100
    print(f"  {name:22s}  MAE = {mae:.2f} MW   MAPE = {mape:.1f} %")

# ---------- 4. What drives the weather correction? (hybrid model) ----------
sample = test.sample(min(5000, len(test)), random_state=42)
sample_ratio = sample[TARGET] / physics(sample)

# 4a. Per-feature permutation importance
imp = permutation_importance(
    model_hybrid, sample[FEATURES], sample_ratio, n_repeats=5, random_state=42
)
importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
print("\nFeature importance for the correction factor (drop in R² when shuffled):")
print(importance.sort_values(ascending=False).round(3).to_string())

plt.figure(figsize=(7, 5))
importance.plot.barh()
plt.xlabel("Importance (drop in R²)")
plt.title("What drives deviation from the physics curve?")
plt.tight_layout()
plt.savefig(FIGS / "feature_importance.png", dpi=150)
plt.show()

# 4b. Grouped weather importance (correlated features shuffled together), on power
weather = [f for f in FEATURES if f != "stw_kn"]
base_r2 = r2_score(sample[TARGET], predict_hybrid(sample))
drops = []
for seed in range(5):
    shuffled = sample.copy()
    idx = np.random.default_rng(seed).permutation(len(shuffled))
    shuffled[weather] = shuffled[weather].values[idx]
    drops.append(base_r2 - r2_score(sample[TARGET], predict_hybrid(shuffled)))
print(f"Grouped weather importance on power (drop in R²): {np.mean(drops):.3f}")

# ---------- 5. Fouling trend: actual vs reference models ----------
df["direct_excess_pct"] = (df[TARGET] / model_direct.predict(df[FEATURES]) - 1) * 100
df["ml_excess_pct"] = (df[TARGET] / predict_hybrid(df) - 1) * 100

daily_direct = df.groupby(df["day"].astype(int))["direct_excess_pct"].median()
daily = df.groupby(df["day"].astype(int))["ml_excess_pct"].median()


def robust_trend(series):
    """Huber fit on out-of-sample days only. Returns (fitted model, % per 30 days)."""
    s = series[series.index >= REF_END]
    h = HuberRegressor().fit(s.index.values.reshape(-1, 1), s.values)
    return h, h.coef_[0] * 30


_, slope_direct = robust_trend(daily_direct)
huber, robust_slope = robust_trend(daily)
print(f"\nRobust out-of-sample trend, pure ML: {slope_direct:+.2f} % per 30 days")
print(f"Robust out-of-sample trend, hybrid:  {robust_slope:+.2f} % per 30 days")
print("(physics + calm filter method: +0.59)")

# ---------- 6. Outlier days: did the hybrid fix them? ----------
oos = daily[daily.index >= REF_END]
oos_direct = daily_direct[daily_direct.index >= REF_END]
outliers = oos[oos.abs() > OUTLIER_PCT]
print(f"\nOutlier days (|excess| > {OUTLIER_PCT} %): "
      f"pure ML = {(oos_direct.abs() > OUTLIER_PCT).sum()}, hybrid = {len(outliers)}")
if len(outliers):
    print(outliers.round(1).to_string())

    out_days = outliers[outliers > 0].index
    mask_out = df["day"].astype(int).isin(out_days)
    normal = df[(df["day"] >= REF_END) & ~mask_out]
    cols = [
        "stw_kn", "power_mw", "Environment_SeaFloorDepth",
        "Weather_WaveHeight", "head_wind_ms", "current_along_ms", "sog_minus_stw_ms",
    ]
    compare = pd.DataFrame({
        "outlier days": df.loc[mask_out, cols].median(),
        "normal days": normal[cols].median(),
        "reference period": ref[cols].median(),
    })
    print("\nMedian values: remaining outlier days vs normal:")
    print(compare.round(2).to_string())

# ---------- 7. Trend plot ----------
in_sample = daily[daily.index < REF_END]
normal_days = oos[oos.abs() <= OUTLIER_PCT]

plt.figure(figsize=(10, 5))
plt.axvspan(0, REF_END, color="grey", alpha=0.15, label="reference period (training)")
plt.scatter(in_sample.index, in_sample.values, s=12, color="grey",
            label="in-sample (not evidence)")
plt.scatter(normal_days.index, normal_days.values, s=12, label="unseen days")
if len(outliers):
    plt.scatter(outliers.index, outliers.values, s=20, color="red", marker="x",
                label=f"outlier days (>{OUTLIER_PCT} %)")
x = oos.index.values
plt.plot(x, huber.predict(x.reshape(-1, 1)), color="black", lw=2,
         label=f"robust trend {robust_slope:+.2f} %/month")
plt.axhline(0, color="grey", ls="--")
plt.xlabel("Day")
plt.ylabel("Excess power vs hybrid reference model (%)")
plt.title("Performance trend on unseen days (hybrid physics-ML model)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "ml_trend.png", dpi=150)
plt.show()

df["power_pred_mw"] = predict_hybrid(df)
df.to_parquet(ROOT / "data" / "processed" / "poseidon_model_v1.parquet")