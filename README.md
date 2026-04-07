# spaflow

`spaflow` is a tool to infer ligand-receptor interaction (LRI) from spatial transcriptomics.

In simple words, it tries to answer this question:

"Where in the tissue are ligand-receptor signals active, and are those signals stronger than we would expect by chance?"

The code is built around `Scanpy` and `Squidpy`.

## What The Source Code Is Doing

The main workflow is:

1. Start with a spatial `AnnData` object that has gene expression and spot coordinates.
2. Build a spatial neighbor graph and a Laplacian matrix from the spot locations.
3. Load and filter ligand-receptor pairs so only supported genes are kept.
4. Use gene expression to set the starting amount of ligand and receptor in each spot.
5. Run a reaction-diffusion style update:
   ligand can spread across nearby spots, ligand and receptor can bind, complexes can break apart, and signals can be produced or degraded.
6. Save the final ligand, receptor, and complex scores back into `adata.obs`.
7. Shuffle spatial coordinates many times to make a null background.
8. Compare the real score to the shuffled scores to get p-values and FDR-adjusted significance calls.
9. Sum ligand-receptor pairs inside the same pathway to get pathway activity scores.

In practice, the package is trying to find spatial communication "hotspots".

## Main Files

[src/spaflow/utils.py](/users/PCON0022/huchen/spaflow/src/spaflow/utils.py) builds the spatial graph, computes the Laplacian matrix, and joins output scores back into `AnnData`.

[src/spaflow/lr.py](/users/PCON0022/huchen/spaflow/src/spaflow/lr.py) loads and filters ligand-receptor databases, keeps pairs that are present in the data, and merges overlapping pathways into larger integrated pathway groups.

[src/spaflow/model.py](/users/PCON0022/huchen/spaflow/src/spaflow/model.py) contains the core simulation. It initializes ligand/receptor levels, updates them over time, and supports several ablation models such as `full`, `no_diffusion`, `no_unbinding`, `no_production`, and `no_degradation`.

[src/spaflow/run.py](/users/PCON0022/huchen/spaflow/src/spaflow/run.py) is the high-level pipeline. It runs the model for each pathway, performs spatial permutation tests, calculates p-values, applies FDR correction, and returns an `AnnData` object with signal and significance columns.

[tutorial.ipynb](/users/PCON0022/huchen/spaflow/tutorial.ipynb) shows an example workflow with different model settings.

## Main Functions

`filter_lr_database(adata)` filters the ligand-receptor database to pairs that are actually supported by the dataset.

`run_spaflow(adata, df_ligrec, ...)` runs the full spatial signaling workflow and adds results to `adata.obs`.

The output usually includes:

- ligand-receptor complex activity scores
- raw p-values
- adjusted p-values
- significant / not significant labels
- pathway activity scores

## Quick Start

Create the environment:

```bash
uv sync
```

Open Python inside the project environment:

```bash
uv run python
```

Basic usage:

```python
import scanpy as sc
import spaflow

adata = sc.read("your_data.h5ad")
df_ligrec = spaflow.filter_lr_database(adata)
adata_out = spaflow.run_spaflow(adata, df_ligrec, n_rounds=10, time_steps=500)
```

## Figure Notebooks

The notebooks in [figures/](/users/PCON0022/huchen/spaflow/figures) are figure-generation notebooks for a manuscript. They mostly read precomputed `.h5ad` and `.csv` files from external paths and then make summary plots.

### Fig. 3

[figures/Fig_3.ipynb](/users/PCON0022/huchen/spaflow/figures/Fig_3.ipynb) focuses on liver / HCC samples and the `CXCL12-CXCR4` signal.

- Fig. 3a shows which significant ligand-receptor hotspots are shared across samples and which ones are sample-specific.
- Fig. 3b shows what kinds of spots make up the `CXCL12-CXCR4` hotspots in responder samples: immune, CAF, or tumor.
- Fig. 3d and Supplementary Fig. 3 test whether immune `CXCL12-CXCR4` hotspots sit near the tumor boundary more often than away from the boundary.
- The spatial plots in the same notebook show the tissue map, spot labels, tumor boundary region, and where `CXCL12-CXCR4` hotspots appear.
- Fig. 3e compares immune spots that are `CXCL12-CXCR4` hotspots against non-hotspots using simple gene-set scores for antigen presentation and T-cell programs.

### Fig. 4

[figures/Fig_4.ipynb](/users/PCON0022/huchen/spaflow/figures/Fig_4.ipynb) focuses on young vs old mouse heart samples.

- Fig. 4a is a volcano plot of pathway activity changes between old and young hearts.
- Fig. 4b is a heatmap showing which pathways increase in specific spatial niches in old hearts, with stars for stronger evidence.
- Fig. 4c shows, for each pathway, which ligand-receptor pairs contribute the most hotspot counts.
- Fig. 4d shows example spatial maps for `Postn-Itgav_Itgb5` and `C3-C3ar1` in young and old samples, together with niche labels and hotspot calls.
- Fig. 4f shows a perturbation summary: if niche 7 is perturbed, which ligand-receptor signals drop in other niches, and how that differs between old and young samples.
- Fig. 4g shows where `Angptl2` and `Pirb` are expressed across single-cell reference cell types.
- Fig. 4h shows which cell types are enriched in selected niches, using a niche-by-cell-type heatmap.

### Fig. 5

[figures/Fig_5.ipynb](/users/PCON0022/huchen/spaflow/figures/Fig_5.ipynb) focuses on lung fibrosis samples and again highlights `CXCL12-CXCR4`.

- Fig. 5a shows spatial maps of tissue annotation, `CXCL12-CXCR4` activity, and significant hotspot locations across control and IPF samples.
- Fig. 5b ranks ligand-receptor pairs that are especially concentrated in lung-fibrosis regions in IPF samples.
- Fig. 5c shows which tissue regions make up the `CXCL12-CXCR4` hotspots in each sample.
- Fig. 5d compares `CXCL12-CXCR4` activity in lung-fibrosis hotspots across Control, IPF upper lung, and IPF lower lung groups.
- Fig. 5e links `CXCL12-CXCR4` activity to fibroblast and immune cell-type abundance using correlation bar plots and p-value heatmaps.
- Fig. 5f shows which single-cell reference cell types express `CXCR4` and `CXCL12`.
- Fig. 5h shows Xenium-based boxplots for `CXCL12+` fibroblasts, `CD4 T`, `CD8 T`, and `B` cells across fibrosis foci, nearby surrounding regions, and control tissue.
