# Data

Should contain CSV files for each respective alloy plus a metadata file.

### Steels

1) Mild steel - `mild.csv`
2) Microalloyed steel - `micr.csv`
3) Maraging steel - `mara.csv`
4) CRNO Steel - `crno.csv`

> [!IMPORTANT]
> The format of each alloy's CSV file should be.
> ```csv
> Strain, Strain Rate (/s), Temperature (deg C), Flow Stress (MPa)
> 0.1   , 0.001           , 850                , 82.3
> ```

# Metadata

The `meta.csv` file contains metadata related to the composition of the alloys being studied.