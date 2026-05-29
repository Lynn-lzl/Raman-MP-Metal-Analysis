import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

def apply_snv(data):
    """Apply Standard Normal Variate (SNV) row-wise."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + 1e-8)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(current_script_dir, "..", "data", "Raman_spectroscopy_data.xlsx")
OUTPUT_FILE = os.path.join(current_script_dir, "..", "data", "Raman_spectroscopy_data_preprocessed.csv")

WAVE_MIN = 400
WAVE_MAX = 1800

SG_WINDOW = 9
SG_POLY = 2
SG_DERIV = 1

def main():
    try:
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        df = pd.read_excel(INPUT_FILE)

        target_cols = []
        metadata_cols = []
        
        for col in df.columns:
            try:
                val = float(col)
                if WAVE_MIN <= val <= WAVE_MAX:
                    target_cols.append(col)
            except (ValueError, TypeError):
                metadata_cols.append(col)

        if not target_cols:
            raise ValueError(f"No columns found in range [{WAVE_MIN}, {WAVE_MAX}].")

        spectra_data = df[target_cols]

        spectra_snv = apply_snv(spectra_data.values)

        spectra_sg = savgol_filter(
            spectra_snv,
            window_length=SG_WINDOW,
            polyorder=SG_POLY,
            deriv=SG_DERIV,
            axis=1
        )

        scaler = StandardScaler()
        spectra_scaled = scaler.fit_transform(spectra_sg)

        processed_spectra_df = pd.DataFrame(
            spectra_scaled, 
            columns=target_cols, 
            index=df.index
        )
        
        final_df = pd.concat([df[metadata_cols], processed_spectra_df], axis=1)
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"Success: Preprocessed data saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()