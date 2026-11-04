from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "poseidon_clean_v1.parquet"
FIGS = ROOT / "figures"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)

# 1. Basic overview
print("Shape:", df.shape)
print("\nColumns:")
for col in df.columns:
    print(" -", col)
print("\nMissing values per column:")
print(df.isna().sum())
print("\nSummary:")
print(df.describe().T)

# 2. Speed-power curve: the core relationship in performance analysis
stw_kn = df["Ship_SpeedThroughWater"] * 1.944          # m/s -> knots
power_mw = df["Propeller_Total_ShaftPower"] / 1e6      # W -> MW
fuel_th = df["Consumer_Total_MomentaryFuel"] * 3.6     # kg/s -> t/h

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(stw_kn, power_mw, s=1, alpha=0.2)
axes[0].set_xlabel("Speed through water (knots)")
axes[0].set_ylabel("Propeller shaft power (MW)")
axes[0].set_title("Speed-power curve")

axes[1].scatter(stw_kn, fuel_th, s=1, alpha=0.2, color="tab:orange")
axes[1].set_xlabel("Speed through water (knots)")
axes[1].set_ylabel("Total fuel (t/h)")
axes[1].set_title("Speed-fuel (includes hotel load)")

plt.tight_layout()
plt.savefig(FIGS / "speed_power_fuel.png", dpi=150)
plt.show()

# 3. Check: is the lower branch caused by missing starboard data?
stbd_missing = df["Propeller_Starboard_ShaftPower"].isna()
print("Rows with starboard missing:", stbd_missing.sum())

plt.figure(figsize=(7, 5))
plt.scatter(stw_kn[~stbd_missing], power_mw[~stbd_missing], s=1, alpha=0.2, label="both propellers")
plt.scatter(stw_kn[stbd_missing], power_mw[stbd_missing], s=1, alpha=0.2, color="red", label="starboard missing")
plt.xlabel("Speed through water (knots)")
plt.ylabel("Propeller shaft power (MW)")
plt.legend(markerscale=10)
plt.tight_layout()
plt.savefig(FIGS / "branch_check.png", dpi=150)
plt.show()