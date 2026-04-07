# spaflow

`spaflow` is a spatial cell-cell communication tool for ligand-receptor interaction analysis in spatial transcriptomics data.

It is designed to answer a practical question:

"Where are ligand-receptor signals active in tissue, and which locations show stronger-than-expected spatial communication?"

The package is built around `Scanpy` and `Squidpy` and uses spatial permutation testing to identify ligand-receptor hotspots.

## Installation

Create the envionment and install from PyPI:

```bash
conda create -n spaflow python=3.10 -y
pip install spaflow
```

If you are working from the repository:

```bash
uv sync
```

## Tutorial

The main tutorial lives in [tutorial.ipynb](tutorial.ipynb). The workflow is:

1. Load a spatial `AnnData` object.
2. Filter the ligand-receptor database to pairs supported by the dataset.
3. Run `spaflow`.
4. Visualize ligand-receptor activity and hotspot calls.

### 1. Load data

```python
import warnings

warnings.filterwarnings("ignore")

import scanpy as sc
from spaflow import filter_lr_database, run_spaflow

adata = sc.read("your_data.h5ad")
```

### 2. Filter the ligand-receptor database

```python
df_ligrec = filter_lr_database(adata)
df_ligrec.head()
```

By default, this keeps supported ligand-receptor pairs for secreted signaling.

### 3. Run SpaFlow

```python
spaflow_adata = run_spaflow(adata, df_ligrec)
```

### 4. Visualize results

The tutorial notebook shows a spatial plot for `TNFSF14-LTBR`:

```python
lr = "TNFSF14-LTBR"

spaflow_adata.uns[f"{lr}_sig_colors"] = ["#DDDDDD", "#EE6677"]

sc.pl.spatial(
    spaflow_adata,
    color=[lr, f"{lr}_sig", "Classes"],
    img_key=None,
    size=1.5,
    frameon=False,
    cmap="Spectral_r",
)
```

In this example, `"Classes"` is an annotation column already present in the tutorial dataset. Replace it with a column from your own `adata.obs` if needed.

## Output

`run_spaflow` returns a copy of the input `AnnData` with additional columns in `adata.obs`.

For each ligand-receptor complex, the output typically includes:

- `<complex>`: ligand-receptor interaction activity
- `<complex>_sig`: ligand-receptor interaction hotspots


These outputs can be used directly in `scanpy.pl.spatial` or downstream statistical summaries.



## Notes
- The notebooks under `figures/` are manuscript-oriented analysis notebooks rather than package onboarding material.
