from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parent.parent
m = pd.read_parquet(ROOT / "data" / "processed" / "poseidon_model_v1.parquet")
FIGS = ROOT / "outputs" / "figures"
REF_END, DEPTH_MIN = 130, 100

deep = m[m["Environment_SeaFloorDepth"] > DEPTH_MIN]
print(f"Deep-water rows: {len(deep)} of {len(m)} ({len(deep) / len(m):.0%})")


def robust_trend(daily, start=REF_END):
    s = daily[daily.index >= start].dropna()
    h = HuberRegressor().fit(s.index.values.reshape(-1, 1), s.values)
    return h.coef_[0] * 30, s, h


# Hybrid model, deep water only
d_ml = deep.groupby(deep["day"].astype(int))["ml_excess_pct"].median()
slope_ml, s_ml, h_ml = robust_trend(d_ml)

# Physics + calm filter, deep water only
calm = deep[(deep["head_wind_ms"].abs() < 5) & (deep["Weather_WaveHeight"] < 1.5)]
d_ph = calm.groupby(calm["day"].astype(int))["excess_power_pct"].median()
slope_ph, _, _ = robust_trend(d_ph, start=0)

print(f"Deep-water trend, hybrid model:       {slope_ml:+.2f} % per 30 days (all water: +0.57)")
print(f"Deep-water trend, physics + calm:     {slope_ph:+.2f} % per 30 days (all water: +0.59)")

plt.figure(figsize=(10, 5))
plt.scatter(s_ml.index, s_ml.values, s=12, label=f"deep water (> {DEPTH_MIN} m), daily median")
x = s_ml.index.values
plt.plot(x, h_ml.predict(x.reshape(-1, 1)), color="black", lw=2,
         label=f"robust trend {slope_ml:+.2f} %/month")
plt.axhline(0, color="grey", ls="--")
plt.xlabel("Day")
plt.ylabel("Excess power vs hybrid reference model (%)")
plt.title("Fouling trend with shallow water excluded")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "deep_water_trend.png", dpi=150)
plt.show()