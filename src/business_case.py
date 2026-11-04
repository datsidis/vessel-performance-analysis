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

# ---------------- Inputs (assumptions: state them in the report) ----------------
TREND_PCT_PER_MONTH = 0.28    # average of physics (+0.59) and hybrid model (+0.57)
FUEL_PRICE_EUR_T = 600        # set to a current bunker price
CLEANING_COST_EUR = 40_000    # assumed in-water hull cleaning + propeller polish
ETA_ELECTRIC = 0.92           # generator -> propeller motor efficiency (assumed)
SAMPLE_MIN = 5
REF_END = 130                 # must match model.py
OUTLIER_PCT = 20              # must match model.py

raw = pd.read_parquet(RAW)
base = pd.read_parquet(BASE)
raw["day"] = raw["index"] * SAMPLE_MIN / 1440

# ---------- 1. SFOC from the ship's own generators ----------
fuel_cols = [f"Consumer_GeneratorEngine{i}_MomentaryFuel" for i in range(1, 6)]
pow_cols = [f"Consumer_GeneratorEngine{i}_ShaftPower" for i in range(1, 6)]
type_cols = [f"Consumer_GeneratorEngine{i}_FuelType" for i in range(1, 6)]
g = raw.dropna(subset=fuel_cols + pow_cols)
gen_fuel = g[fuel_cols].sum(axis=1)          # kg/s
gen_power = g[pow_cols].sum(axis=1)          # W
ok = gen_power > 1e6                          # ignore near-idle rows
sfoc = gen_fuel[ok].sum() / gen_power[ok].sum() * 3.6e9   # g/kWh
print(f"SFOC from data: {sfoc:.0f} g/kWh (typical medium-speed diesel: ~180-220)")

# CO2 factor weighted by the fuel actually burned (IMO factors: HFO 3.114, MGO/MDO 3.206)
burned = {}
for f_col, t_col in zip(fuel_cols, type_cols):
    for ftype, kg in g.groupby(t_col)[f_col].sum().items():
        burned[ftype] = burned.get(ftype, 0) + kg
factors = {ft: (3.206 if str(ft).startswith("DM") else 3.114) for ft in burned}
co2_factor = sum(burned[ft] * factors[ft] for ft in burned) / sum(burned.values())
shares = {ft: f"{kg / sum(burned.values()):.0%}" for ft, kg in burned.items()}
print(f"Fuel mix burned: {shares} -> CO2 factor {co2_factor:.3f} t CO2 / t fuel")

# ---------- 2. Annual propulsion energy ----------
stw_kn = raw["Ship_SpeedThroughWater"] * 1.944
sailing = stw_kn > 8
sailing_hours = sailing.sum() * SAMPLE_MIN / 60
mean_prop_mw = base["power_mw"].mean()
prop_mwh_year = mean_prop_mw * sailing_hours
print(f"Sailing hours/year: {sailing_hours:.0f}, mean propulsion power: {mean_prop_mw:.1f} MW")
print(f"Propulsion energy/year: {prop_mwh_year:,.0f} MWh")

# ---------- 3. What 1 % extra propulsion power costs per month ----------
fuel_t_per_prop_mwh = (sfoc / 1000) / ETA_ELECTRIC          # t fuel per MWh at the propeller
fuel_per_pct_month = 0.01 * (prop_mwh_year / 12) * fuel_t_per_prop_mwh
cost_per_pct_month = fuel_per_pct_month * FUEL_PRICE_EUR_T
print(f"1 % extra power = {fuel_per_pct_month:.1f} t fuel/month = EUR {cost_per_pct_month:,.0f}/month")

# ---------- 4. Linear fouling model: optimal cleaning interval ----------
k = cost_per_pct_month * TREND_PCT_PER_MONTH                 # extra cost rate grows by k each month
t_opt = np.sqrt(2 * CLEANING_COST_EUR / k)                   # minimises average cost per month
no_clean_fuel_12m = fuel_per_pct_month * TREND_PCT_PER_MONTH * 12 ** 2 / 2

print("\n--- A. Linear fouling model ---")
print(f"Without cleaning, year 1 extra fuel: {no_clean_fuel_12m:,.0f} t "
      f"= EUR {no_clean_fuel_12m * FUEL_PRICE_EUR_T:,.0f} "
      f"= {no_clean_fuel_12m * co2_factor:,.0f} t CO2")
print(f"Optimal cleaning interval: every {t_opt:.1f} months")

print("\nSensitivity: optimal cleaning interval (months)")
trends = [0.3, TREND_PCT_PER_MONTH, 0.9]
costs = [20_000, 40_000, 80_000]
table = pd.DataFrame(
    [[np.sqrt(2 * c / (cost_per_pct_month * t)) for c in costs] for t in trends],
    index=[f"trend {t}%/month" for t in trends],
    columns=[f"cleaning EUR {c // 1000}k" for c in costs],
)
print(table.round(1).to_string())

# ---------- 5. Measured extra fuel, using the hybrid reference model ----------
m = pd.read_parquet(MODEL)
m["day_int"] = m["day"].astype(int)
oos = m[m["day"] >= REF_END].copy()

# Drop the shallow-water outlier days (outside the model's reference data)
daily_med = oos.groupby("day_int")["ml_excess_pct"].median()
bad_days = daily_med[daily_med.abs() > OUTLIER_PCT].index
oos = oos[~oos["day_int"].isin(bad_days)]

# Excess energy per 5-minute row (negative rows count too: net effect)
oos["excess_mwh"] = (oos["power_mw"] - oos["power_pred_mw"]) * SAMPLE_MIN / 60

# Clean data covers only part of the sailing time: scale up to all sailing rows
sailing_rows_period = (sailing & (raw["day"] >= REF_END)).sum()
scale = sailing_rows_period / len(oos)
print(f"\n--- B. Measured with the hybrid model (days {REF_END}-365) ---")
print(f"Clean rows cover {1 / scale:.0%} of sailing time in this period; results scaled up x{scale:.2f}")

oos["block"] = (oos["day"] // 30).astype(int)
monthly = oos.groupby("block").agg(
    start_day=("day", "min"),
    mean_excess_pct=("ml_excess_pct", "mean"),
    excess_mwh=("excess_mwh", "sum"),
)
monthly["excess_mwh"] *= scale
monthly["fuel_t"] = monthly["excess_mwh"] * fuel_t_per_prop_mwh
monthly["cost_eur"] = monthly["fuel_t"] * FUEL_PRICE_EUR_T
monthly["co2_t"] = monthly["fuel_t"] * co2_factor
print(monthly.round({"start_day": 0, "mean_excess_pct": 1, "excess_mwh": 0,
                     "fuel_t": 1, "cost_eur": 0, "co2_t": 0}).to_string(index=False))

total = monthly[["fuel_t", "cost_eur", "co2_t"]].sum()
days_covered = 365 - REF_END
print(f"\nMeasured extra fuel over {days_covered} days: {total['fuel_t']:,.0f} t "
      f"= EUR {total['cost_eur']:,.0f} = {total['co2_t']:,.0f} t CO2")
print("Note: measured against the reference-period hull (days 0-130), not a freshly "
      "cleaned hull, so this is a lower bound on the true fouling cost.")

# ---------- 6. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

months = np.linspace(0, 24, 500)
axes[0].plot(months, k * months ** 2 / 2 / 1000, lw=2,
             label="cumulative extra fuel cost since cleaning")
axes[0].axhline(CLEANING_COST_EUR / 1000, color="grey", ls="--", label="cleaning cost")
axes[0].axvline(t_opt, color="black", ls=":", label=f"optimal interval: {t_opt:.1f} months")
axes[0].set_xlabel("Months since last cleaning")
axes[0].set_ylabel("EUR (thousands)")
axes[0].set_title("A. Linear model: when does cleaning pay off?")
axes[0].legend()

labels = [f"day {int(d)}" for d in monthly["start_day"]]
colors = ["tab:red" if c > 0 else "tab:blue" for c in monthly["cost_eur"]]
axes[1].bar(labels, monthly["cost_eur"] / 1000, color=colors)
axes[1].axhline(0, color="grey", lw=1)
axes[1].set_ylabel("Extra fuel cost (EUR thousands)")
axes[1].set_title("B. Measured: extra fuel cost per 30 days (hybrid model)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(FIGS / "business_case.png", dpi=150)
plt.show()

# Cleaning or season? Check how fast the drop happens and whether temperature follows the hump
m_day = m[m["day"] >= REF_END].groupby("day_int").agg(
    excess_pct=("ml_excess_pct", "median"),
    air_temp=("Weather_Temperature2M", "median"),
    depth_m=("Environment_SeaFloorDepth", "median"),
    stw_kn=("stw_kn", "median"),
)
print("\nDaily values around the drop (days 295-345):")
print(m_day.loc[295:345].round(1).to_string())

blocks = m_day.groupby(m_day.index // 30 * 30).median()
print("\n30-day medians: does temperature follow the hump?")
print(blocks.round(1).to_string())