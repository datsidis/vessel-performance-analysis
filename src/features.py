from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "processed" / "poseidon_clean_v1.parquet"
OUT = ROOT / "data" / "processed" / "poseidon_features_v1.parquet"
FIGS = ROOT / "outputs" / "figures"

df = pd.read_parquet(CLEAN)


def relative_angle(direction_from, heading):
    """0 deg = coming from straight ahead, 180 deg = from straight behind."""
    diff = (direction_from - heading) % 360
    return np.where(diff > 180, 360 - diff, diff)


# Wind relative to the ship
df["rel_wind_deg"] = relative_angle(df["Weather_WindDirection10M"], df["Ship_Heading"])
df["head_wind_ms"] = df["Weather_WindSpeed10M"] * np.cos(np.radians(df["rel_wind_deg"]))  # + = head wind

# Waves relative to the ship
df["rel_wave_deg"] = relative_angle(df["Weather_WaveDirection"], df["Ship_Heading"])

# Current along the ship's heading (assumes direction = where the current flows TO)
diff_cur = np.radians(df["Weather_OceanCurrentDirection"] - df["Ship_Heading"])
df["current_along_ms"] = df["Weather_OceanCurrentVelocity"] * np.cos(diff_cur)  # + = pushing the ship

# Check the current convention: SOG - STW should roughly equal the current along track
df["sog_minus_stw_ms"] = df["Ship_SpeedOverGround"] - df["Ship_SpeedThroughWater"]
corr = df["current_along_ms"].corr(df["sog_minus_stw_ms"])
print(f"Correlation current_along vs (SOG - STW): {corr:.2f}")
print("Positive = convention correct. Negative = flip the sign.")

df.to_parquet(OUT)
print(f"Saved {len(df)} rows with features")

# Plot: does head wind push the speed-power curve up?
plt.figure(figsize=(8, 5))
sc = plt.scatter(df["stw_kn"], df["power_mw"], c=df["head_wind_ms"],
                 cmap="coolwarm", s=1, alpha=0.4, vmin=-12, vmax=12)
plt.colorbar(sc, label="Head wind component (m/s), red = head, blue = tail")
plt.xlabel("Speed through water (knots)")
plt.ylabel("Propeller shaft power (MW)")
plt.title("Speed-power coloured by head wind")
plt.tight_layout()
plt.savefig(FIGS / "speed_power_headwind.png", dpi=150)
plt.show()