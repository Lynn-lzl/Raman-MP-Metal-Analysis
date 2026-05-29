# Raman Spectroscopy Modeling

This project trains and evaluates machine learning models for Raman spectroscopy data.

## Setup

Create and activate a conda environment with Python 3.10, then install the required packages:

```bash
conda create -n raman python=3.10
conda activate raman
pip install -r requirements.txt
```

## Data

Place the raw data file at:

```text
data/Raman_spectroscopy_data.xlsx
```

## Preprocessing

Run the preprocessing script first:

```bash
python data_preprocessing/spectral_preprocessing.py
```

This generates the preprocessed CSV file used by the modeling pipeline.

## Run

After preprocessing, run the main experiment:

```bash
python main.py
```

The main script loads the preprocessed data, trains the models, and exports the results.
