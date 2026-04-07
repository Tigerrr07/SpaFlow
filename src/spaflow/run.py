import os
from pathlib import Path
from .utils import get_laplacian_mtx, concat_lr_adata
from .model import (
    grd_update,
    initial_concentration,
    pathway_activity_agg_with_plots,
)
from statsmodels.stats.multitest import multipletests
from typing import Optional
import numpy as np
import pandas as pd

def base_spaflow(adata, df_ligrec, pathway_name, time_steps=500, res_dir=None, model="full"):
    # print(f"Running {model} model")
    # Filter ligand receptor pairs only belong to this pathway
    lr_info = df_ligrec[df_ligrec['integrated_pathway'] == pathway_name]
    adata = adata.copy()
    

    concentrations = initial_concentration(adata, lr_info)

    # Access components
    ligand_concentration_df = concentrations['ligand_concentration']
    receptor_concentration_df = concentrations['receptor_concentration']
    complex_concentration_df = concentrations['complex_concentration']

    lap_mtx = adata.obsp['laplacian_matrix']

    backbone_L = ligand_concentration_df.values.copy()
    backbone_R = receptor_concentration_df.values.copy()
    backbone_C = complex_concentration_df.values.copy()

    ligand_step_list, receptor_step_list, complex_step_list, new_ligand_data, _ = grd_update(
        backbone_L,
        backbone_R,
        backbone_C,
        lr_info,
        lap_mtx,
        model=model,
        T=time_steps,
        verbose=False,
    )

    # new grd adata, only with ligand/receptor/complex concentration
    grd_adata = concat_lr_adata(adata, ligand_step_list, receptor_step_list, complex_step_list, lr_info)

    if res_dir is not None:
        max_parts = 5
        pathway_name_parts = pathway_name.split("_")
        # If the length exceeds 5, only the first 5 are retained to form the new path name.
        if len(pathway_name_parts) > max_parts:
            short_name = "_".join(pathway_name_parts[:max_parts])
        else:
            short_name = pathway_name
        
        p_res_dir = Path(res_dir) / short_name
        os.makedirs(p_res_dir, exist_ok=True)
        
        plot_dynamics_curve(grd_adata, ligand_step_list, receptor_step_list, complex_step_list, lr_info, show_plot=True, save_dir=p_res_dir)
    
        # plot_lrc_spatial(
        #     adata=grd_adata,
        #     lr_info=lr_info,
        #     steps=[0, 250, 500],
        #     save_dir=p_res_dir,
        #     show_plot=False,
        #     cmap='RdBu_r'
        # )
    
        # plot_ligand_diffusion(
        #     ligand_step_list=ligand_step_list,
        #     new_ligand_data=new_ligand_data,
        #     adata=grd_adata,
        #     lr_info=lr_info,
        #     steps_to_plot=[0, 1, 5, 10],  # Specific steps to plot, or None for all
        #     save_dir=p_res_dir,
        #     show_plot=False,  # Set to True to display plots interactively
        #     spot_size=5
        # )

    return grd_adata

def permute_lr_significance_spatial(
    adata,
    df_ligrec: pd.DataFrame,
    pathway_name: str,
    perm_indices_list,
    spatial_coord_key: str = 'spatial',
    # n_rounds=10,
    store_rounds=False,
    res_dir = None,
    time_steps=500,
    model="full"
) -> Optional[pd.DataFrame]:
    """
    Permute spatial coordinates to calculate significance of LR scores.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with spatial coordinates
    df_ligrec : pd.DataFrame
        DataFrame with ligand-receptor information.
    pathway_name : str
        Pathway name (e.g., "IL6_LIFR") to test
    spatial_coord_key : str
        Key in adata.obsm where spatial coordinates are stored (default: 'spatial')
    perm_size : int
        Number of permutations for null distribution
    inplace : bool
        If True, add results to adata.obs. If False, return DataFrame
    store_rounds : bool
        If True, store scores from each permutation round in adata.obs
        
    Returns:
    --------
    pd.DataFrame or None
        Results with observed scores, p-values, and significance flags
        (only returned if inplace=False)
    """
    adata = adata.copy()
    original_coords = adata.obsm[spatial_coord_key].copy()
    n_cells = adata.n_obs
        
    # print(f"Testing significance for pathway: {pathway_name} with {n_rounds} rounds")
    
    # Calculate observed scores using original coordinates
    # print("Calculating observed scores...")
    
    # Ensure original spatial connectivity is calculated
    adata = get_laplacian_mtx(adata,
        spatial_key='spatial',
        coord_type='grid',
        normalization=True,
        n_neighs=6
    )

    grd_adata = base_spaflow(adata, df_ligrec=df_ligrec, pathway_name=pathway_name, res_dir=res_dir, model=model)

    # For each integrated pathway
    lr_info = df_ligrec[df_ligrec['integrated_pathway'] == pathway_name]
    # With pathway aggergation
    for p_name in lr_info['pathway'].unique():
        # print(p_name)
        grd_adata = pathway_activity_agg_with_plots(grd_adata, lr_info, time_step=time_steps, pathway_name=p_name, res_dir = res_dir,
                                       plt_show=False, cmap='coolwarm', img_key=None, spot_size=150)
    
    # Extract observed scores
    observed_complex_scores = extract_scores_from_grd_adata(grd_adata, pathway_name, df_ligrec)
    
    # Generate null distribution for each complex
    # print("Generating null distribution...")
    all_null_complex_scores = {}
    
    for round_i, perm_indices in enumerate(perm_indices_list):
        # print(f"Round {round_i + 1}/{len(perm_indices_list)}")
        
        # Generate permutation indices for spatial coordinates
        # perm_indices = np.random.permutation(n_cells)

        # Calculate null scores for this round using permuted coordinates
        null_complex_scores = calculate_null_scores_with_spatial_permutation(
            adata, df_ligrec, pathway_name,
            original_coords, perm_indices, spatial_coord_key, model=model
        )
        
        # Store null scores for each complex
        for complex_name, null_scores in null_complex_scores.items():
            if complex_name not in all_null_complex_scores:
                all_null_complex_scores[complex_name] = []
            all_null_complex_scores[complex_name].append(null_scores)
            
            # Store individual round scores in grd_adata.obs if requested
            if store_rounds:
                round_key = f'{complex_name}_spatial_perm_round_{round_i+1}'
                grd_adata.obs[round_key] = null_scores.flatten()
                # print(f"  Stored round {round_i+1} scores as: {round_key}")
    
    # Combine all null scores for each complex
    # print("Calculating p-values for each complex...")
    results_dict = {}
    
    for complex_name, observed_scores in observed_complex_scores.items():
        # Combine null scores across rounds
        all_null_scores = np.concatenate(all_null_complex_scores[complex_name], axis=1)
        
        
        # Calculate p-values
        p_values = calculate_pvalues_rank_based(observed_scores, all_null_scores)
        
        # Store results with 'spatial' prefix to distinguish from graph permutation
        results_dict[f'{complex_name}_observed'] = observed_scores
        del grd_adata.obs[f"grd_{complex_name}_step500"] # remove duplicate one

        results_dict[f'{complex_name}_pvalue'] = p_values
        results_dict[f'{complex_name}_p_sig_05'] = p_values < 0.05

    # Add to adata.obs
    results_df = pd.DataFrame(results_dict, index=grd_adata.obs.index)
    
    # join in one shot
    grd_adata.obs = grd_adata.obs.join(results_df)
    
    # print(f"Spatial permutation results added to grd_adata.obs:")
    # for key in results_dict.keys():
    #     print(f"  - {key}")
    
    # Print summary for each complex
    total_sig_05 = 0
    # print(f"\nSpatial permutation summary for pathway {pathway_name}:")
    for complex_name in observed_complex_scores.keys():
        n_sig_05 = np.sum(results_dict[f'{complex_name}_p_sig_05'])
        # print(f"  {complex_name}: {n_sig_05}/{n_cells} cells (p<0.05)")

    return grd_adata


def calculate_null_scores_with_spatial_permutation(
    adata, 
    df_ligrec: pd.DataFrame,
    pathway_name: str,
    original_coords: np.ndarray,
    perm_indices: np.ndarray,
    spatial_coord_key: str,
    model: str = "full",
) -> dict:
    """
    Calculate null scores using permuted spatial coordinates.
    
    Returns a dictionary with complex names as keys and null scores as values.
    """
    
    # Create temporary adata copy
    adata_temp = adata.copy()
    
    # Permute spatial coordinates
    permuted_coords = original_coords[perm_indices, :]
    adata_temp.obsm[spatial_coord_key] = permuted_coords
    
    # Recalculate spatial connectivity from permuted coordinates
    adata_temp = get_laplacian_mtx(adata_temp,
        spatial_key='spatial',
        coord_type='grid',
        normalization=True,
        n_neighs=6
    )
    
    # Run spaflow with permuted coordinates and updated connectivity
    null_grd_adata = base_spaflow(
        adata_temp,
        df_ligrec=df_ligrec,
        pathway_name=pathway_name,
        model=model,
    )
    
    # Extract null scores for all complexes
    null_complex_scores = extract_scores_from_grd_adata(null_grd_adata, pathway_name, df_ligrec)
    
    # Reshape each complex's scores
    reshaped_null_scores = {}
    for complex_name, scores in null_complex_scores.items():
        reshaped_null_scores[complex_name] = scores.reshape(-1, 1)
    
    return reshaped_null_scores


def calculate_pvalues_rank_based(observed_scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """
    Parameters:
    -----------
    observed_scores : np.ndarray
        Observed scores for each cell (n_cells,)
    null_scores : np.ndarray  
        Null distribution scores (n_cells, n_permutations)
        
    Returns:
    --------
    np.ndarray
        P-values for each cell
    """
    n_cells = len(observed_scores)
    p_values = np.zeros(n_cells)
    
    for i in range(n_cells):
        observed = observed_scores[i]
        null_dist = null_scores[i, :]
        
        sorted_null = np.sort(null_dist)
        

        rank = np.searchsorted(sorted_null, observed, side='left')
        
        # Convert rank to p-value: p = 1 - rank/n_null
        # This gives P(null >= observed) - right-tailed test
        p_values[i] = 1.0 - rank / len(null_dist)
        
    return p_values

def extract_scores_from_grd_adata(grd_adata, pathway_name: str, df_ligrec: pd.DataFrame) -> dict:
    """
    Extract scores from the output of base_spaflow for all L-R pairs in a pathway.
    
    Returns a dictionary with complex names as keys and scores as values.
    """
    
    # Get L-R pairs for this pathway
    lr_info = df_ligrec[df_ligrec['integrated_pathway'] == pathway_name]
    
    # Extract scores for each complex
    complex_scores = {}
    
    for _, row in lr_info.iterrows():
        complex_name = row['complex']
        complex_scores[complex_name] = grd_adata.obs[f"grd_{complex_name}_step500"].values

    return complex_scores


def run_spaflow(adata, df_ligrec, fdr_method="fdr_bh", alpha=0.05, n_rounds=10, res_dir=None, time_steps=500, model="full"):
    """
    Apply FDR correction per cell across all interaction p-values
    already stored in adata.obs by permute_lr_significance_spatial.
    """
    # print(f"{len(df_ligrec['integrated_pathway'].unique())} l-r intergrated pathways need to be processed")

    adata = adata.copy()

    n_cells = adata.shape[0]

    perm_indices_list = []
    for i in range(n_rounds):
        perm_indices = np.random.permutation(n_cells)
        perm_indices_list.append(perm_indices)
        
    for p in df_ligrec['integrated_pathway'].unique():
        adata = permute_lr_significance_spatial(adata, df_ligrec=df_ligrec, pathway_name=p, perm_indices_list=perm_indices_list, res_dir=res_dir, time_steps=time_steps, model=model)
        # print("======================\n")
    
    # collect all *_pvalue columns
    pval_keys = [k for k in adata.obs.columns if k.endswith("_pvalue")]
    if not pval_keys:
        raise ValueError("No *_pvalue columns found. Run permute_lr_significance_spatial first.")
    
    P = np.column_stack([adata.obs[k].values for k in pval_keys])
    Q = np.empty_like(P, dtype=float)

    for i in range(P.shape[0]):  # per cell
        _, adj, _, _ = multipletests(P[i], method=fdr_method, alpha=alpha)
        Q[i] = adj


    new_cols = {}
    for j, key in enumerate(pval_keys):
        base = key.replace("_pvalue", "")
        padj_col = f"{base}_padj"
        sig_col = f"{base}_padj_sig_05"
        new_cols[padj_col] = Q[:, j]
        new_cols[sig_col] = Q[:, j] < alpha
        
        # existing raw significance column
        raw_sig_col = f"{base}_p_sig_05"
        if raw_sig_col in adata.obs:
            raw_sig_cells = adata.obs.index[adata.obs[raw_sig_col]].tolist()
        else:
            raw_sig_cells = []
        
        padj_sig_cells = adata.obs.index[new_cols[sig_col]].tolist()
        
        # print(f"{base}: raw_p={len(raw_sig_cells)}, padj={len(padj_sig_cells)}")
    
    adata.obs = pd.concat([adata.obs, pd.DataFrame(new_cols, index=adata.obs.index)], axis=1)

    rename_map = {}
    for col in adata.obs.columns:
        if col.endswith("_observed"):
            rename_map[col] = col[:-len("_observed")]
        elif col.endswith("_p_sig_05"):
            rename_map[col] = f"{col[:-len('_p_sig_05')]}_sig"

    if rename_map:
        adata.obs = adata.obs.rename(columns=rename_map)


    return adata
