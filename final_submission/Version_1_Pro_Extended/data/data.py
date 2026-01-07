import xarray as xr
import pandas as pd

# Load GRIB file
ds = xr.open_dataset("data.grib", engine="cfgrib")

# Convert to a DataFrame
df = ds.to_dataframe().reset_index()

# Save as CSV
df.to_csv("output.csv", index=False)

print("Saved output.csv")
