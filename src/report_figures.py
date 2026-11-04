"""Extra figures for the README: cleaning before/after, shallow-water timeline, depth effect."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "CPS_Poseidon.parquet"
BASE = ROOT / "data" / "processed" / "poseidon_baseline_v1.parquet"
MODEL = ROOT / "data" / "processed" / "poseidon_model_v1.parquet"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

REF_END = 130  # must match model.py

raw = pd.read_parquet(RAW)
base = pd.read_parquet(BASE)
m = pd.read_parquet(MODEL)

# ---------- 1. Cleaning: before vs after ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
axes[0].scatter(raw["Ship_SpeedThroughWater"] * 1.944,
                raw["Propeller_Total_ShaftPower"] / 1e6,
                s=1, alpha=0.2, color="grey")
axes[0].set_title(f"Raw data ({len(raw):,} rows)")
axes[1].scatter(base["stw_kn"], base["power_mw"], s=1, alpha=0.2)
axes[1].set_title(f"After cleaning ({len(base):,} rows)")
for ax in axes:
    ax.set_xlabel("Speed through water (knots)")
axes[0].set_ylabel("Propeller shaft power (MW)")
fig.suptitle("Cleaning removes sensor dropouts, port time and the false 'one-propeller' branch")
plt.tight_layout()
plt.savefig(FIGS / "cleaning_before_after.png", dpi=150)
plt.show()

# ---------- 2. Shallow-water timeline (headline figure) ----------
oos = m[m["day"] >= REF_END].copy()
daily = oos.groupby(oos["day"].astype(int)).agg(
    excess=("ml_excess_pct", "median"),
    depth=("Environment_SeaFloorDepth", "median"),
)
weekly = daily.rolling(7, center=True, min_periods=3).median()

fig, ax1 = plt.subplots(figsize=(11, 5))
l1 = ax1.plot(weekly.index, weekly["excess"], color="tab:red", lw=2,
              label="excess power, 7-day median")
ax1.axhline(0, color="grey", ls="--", lw=1)
ax1.set_xlabel("Day")
ax1.set_ylabel("Excess power vs reference model (%)", color="tab:red")

ax2 = ax1.twinx()
l2 = ax2.plot(weekly.index, weekly["depth"], color="tab:blue", lw=2,
              label="sea floor depth, 7-day median (axis inverted)")
ax2.set_ylabel("Sea floor depth (m), shallow = up", color="tab:blue")
ax2.invert_yaxis()

lines = l1 + l2
ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
plt.title("Excess power rises when the ship operates in shallow water")
plt.tight_layout()
plt.savefig(FIGS / "shallow_water_timeline.png", dpi=150)
plt.show()

# ---------- 3. Excess power by depth band ----------
bins = [0, 25, 50, 100, 200, np.inf]
labels = ["< 25 m", "25-50 m", "50-100 m", "100-200 m", "> 200 m"]
oos["depth_band"] = pd.cut(oos["Environment_SeaFloorDepth"], bins=bins, labels=labels)
groups = [oos.loc[oos["depth_band"] == lab, "ml_excess_pct"].dropna() for lab in labels]

print("Median excess power by depth band (unseen days):")
for lab, grp in zip(labels, groups):
    print(f"  {lab:10s}  median {grp.median():+5.1f} %   (n = {len(grp):,})")

plt.figure(figsize=(9, 5))
plt.boxplot(groups, showfliers=False)
plt.xticks(range(1, len(labels) + 1), labels)
plt.axhline(0, color="grey", ls="--", lw=1)
for i, grp in enumerate(groups, start=1):
    plt.text(i, plt.ylim()[1] * 0.92, f"n={len(grp):,}", ha="center", fontsize=9)
plt.xlabel("Sea floor depth")
plt.ylabel("Excess power vs reference model (%)")
plt.title("The shallower the water, the more power the ship needs")
plt.tight_layout()
plt.savefig(FIGS / "depth_effect.png", dpi=150)
plt.show()