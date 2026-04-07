import pandas as pd
import numpy as np
import scipy.sparse as ss
from tqdm import tqdm
from joblib import Parallel, delayed
import contextlib
import joblib
import os
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

# Simple snippet copied from https://stackoverflow.com/a/58936697/5133167
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def initial_concentration(adata, lr_info, heteromeric_rule = 'min', heteromeric_delimiter = '_', l_name = 'ligand', r_name = 'receptor', c_name = 'complex'):
    """
    Initialize concentration matrices for ligands, receptors, and complexes.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing gene expression data.
    lr_info : pd.DataFrame
        DataFrame containing ligand-receptor pairs information.
    initial_zero : bool, default=False
        If True, initialize all concentrations to zero instead of actual ligand/receptor expression values.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'ligand_concentration': DataFrame of ligand concentrations
        - 'receptor_concentration': DataFrame of receptor concentrations
        - 'complex_concentration': DataFrame of complex concentrations
    """

    avail_ligs = pd.unique(lr_info[l_name]).tolist()
    avail_recs = pd.unique(lr_info[r_name]).tolist()
    
    interaction_list = lr_info[c_name].tolist()

    # Get unique ligands and receptors while preserving order
    # unique_ligands = pd.unique(lr_info['ligand']).tolist()
    # unique_receptors = pd.unique(lr_info['receptor']).tolist()
    # interaction_list = lr_info.index.tolist()

    
    # Initialize concentration matrices

    ligand_exp = np.zeros((adata.n_obs, len(avail_ligs)))
    receptor_exp = np.zeros((adata.n_obs, len(avail_recs)))

    for i in range(len(avail_ligs)):
        tmp_lig = avail_ligs[i]
        lig_genes = tmp_lig.split(heteromeric_delimiter)
        if heteromeric_rule == 'min':
            ligand_exp[:,i] = adata[:, lig_genes].X.toarray().min(axis=1)[:]
        elif heteromeric_rule == 'ave':
            ligand_exp[:,i] = adata[:, lig_genes].X.toarray().mean(axis=1)[:]

    for i in range(len(avail_recs)):
        tmp_rec = avail_recs[i]
        rec_genes = tmp_rec.split(heteromeric_delimiter)
        if heteromeric_rule == 'min':
            receptor_exp[:,i] = adata[:, rec_genes].X.toarray().min(axis=1)[:]
        elif heteromeric_rule == 'ave':
            receptor_exp[:,i] = adata[:, rec_genes].X.toarray().mean(axis=1)[:]

    # Create DataFrames
    ligand_concentration = pd.DataFrame(
        ligand_exp,
        index=adata.obs_names,
        columns=avail_ligs
    )
    
    receptor_concentration = pd.DataFrame(
        receptor_exp,
        index=adata.obs_names,
        columns=avail_recs
    )
    
    complex_concentration = pd.DataFrame(
        0.,
        index=adata.obs_names,
        columns=interaction_list
    )
    
    # Return everything in a dictionary
    return {
        'ligand_concentration': ligand_concentration,
        'receptor_concentration': receptor_concentration,
        'complex_concentration': complex_concentration,
    }

    

def lr_info_to_mapping(lr_info):    
    # Extract unique ligands and receptors while preserving order
    # lr_info need reset index
    unique_ligands = pd.unique(lr_info['ligand']).tolist()
    unique_receptors = pd.unique(lr_info['receptor']).tolist()
    
    # Create lr_mapping directly from lr_info
    lr_mapping = pd.DataFrame(0, index=unique_receptors, columns=unique_ligands)
    for idx, row in lr_info.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        lr_mapping.loc[receptor, ligand] = 1

    return lr_mapping, unique_ligands, unique_receptors


VALID_GRD_MODELS = {
    'full',
    'no_diffusion',
    'no_unbinding',
    'no_production',
    'no_degradation',
}


def _compute_grd_derivatives(
    model,
    L,
    R,
    C,
    diffusion_L,
    binding_L,
    binding_R,
    binding_C,
    unbinding_L,
    unbinding_R,
    unbinding_C,
    S_L_feedback,
    S_R_feedback,
    lambda_L,
    lambda_R,
    delta_L,
    delta_R,
    delta_C,
):
    if model == 'full':
        dL_dt = diffusion_L + lambda_L * S_L_feedback - binding_L + unbinding_L - delta_L * L
        dR_dt = lambda_R * S_R_feedback - binding_R + unbinding_R - delta_R * R
        dC_dt = binding_C - unbinding_C - delta_C * C
    elif model == 'no_diffusion':
        dL_dt = lambda_L * S_L_feedback - binding_L + unbinding_L - delta_L * L
        dR_dt = lambda_R * S_R_feedback - binding_R + unbinding_R - delta_R * R
        dC_dt = binding_C - unbinding_C - delta_C * C
    elif model == 'no_unbinding':
        dL_dt = diffusion_L + lambda_L * S_L_feedback - binding_L - delta_L * L
        dR_dt = lambda_R * S_R_feedback - binding_R - delta_R * R
        dC_dt = binding_C - delta_C * C
    elif model == 'no_production':
        dL_dt = diffusion_L - binding_L + unbinding_L - delta_L * L
        dR_dt = unbinding_R - binding_R - delta_R * R
        dC_dt = binding_C - unbinding_C - delta_C * C
    elif model == 'no_degradation':
        dL_dt = diffusion_L + lambda_L * S_L_feedback - binding_L + unbinding_L
        dR_dt = lambda_R * S_R_feedback - binding_R + unbinding_R
        dC_dt = binding_C - unbinding_C
    else:
        supported_models = ', '.join(sorted(VALID_GRD_MODELS))
        raise ValueError(f"Unsupported model '{model}'. Expected one of: {supported_models}")

    return dL_dt, dR_dt, dC_dt


def grd_update(
    L,
    R,
    C,
    lr_info,
    lap_mtx,
    model='full',
    D_L=1,
    dt=0.01,
    T=10,
    k_on=1.0,
    k_off=1.0,
    lambda_L=0.1,
    lambda_R=0.1,
    delta_L=0.1,
    delta_R=0.1,
    delta_C=0.1,
    alpha=1.0,
    beta=1.0,
    verbose=False,
    tol=1e-2,
):
    """
    GRD model iterative update with bidirectional competition mechanism between ligands and receptors.

    Parameters
    ----------
    model : str, default='full'
        Model variant. Supported values: 'full', 'no_diffusion', 'no_unbinding',
        'no_production', and 'no_degradation'.
    """
    if model not in VALID_GRD_MODELS:
        supported_models = ', '.join(sorted(VALID_GRD_MODELS))
        raise ValueError(f"Unsupported model '{model}'. Expected one of: {supported_models}")

    lr_mapping, unique_ligands, _ = lr_info_to_mapping(lr_info)

    initial_zero_mask = (L == 0).copy()
    new_diffusion_ligand = []

    base_L = L.copy()
    base_R = R.copy()

    ligand_step_list = [L.copy()]
    receptor_step_list = [R.copy()]
    complex_step_list = [C.copy()]

    receptor_to_ligands = {}
    ligand_to_receptors = {}
    for _, row in lr_info.iterrows():
        l_idx = int(lr_mapping.columns.get_indexer([row['ligand']])[0])
        r_idx = int(lr_mapping.index.get_indexer([row['receptor']])[0])
        receptor_to_ligands.setdefault(r_idx, []).append(l_idx)
        ligand_to_receptors.setdefault(l_idx, []).append(r_idx)

    _, n_ligands = L.shape
    _, n_receptors = R.shape

    mask_rec = {}
    for l_idx, rec_list in ligand_to_receptors.items():
        mask = np.zeros(n_receptors, dtype=np.float32)
        mask[rec_list] = 1.0
        mask_rec[l_idx] = mask

    mask_lig = {}
    for r_idx, lig_list in receptor_to_ligands.items():
        mask = np.zeros(n_ligands, dtype=np.float32)
        mask[lig_list] = 1.0
        mask_lig[r_idx] = mask

    binding_L = np.zeros_like(L)
    binding_R = np.zeros_like(R)
    binding_C = np.zeros_like(C)
    unbinding_L = np.zeros_like(L)
    unbinding_R = np.zeros_like(R)

    convergence_steps = np.full(n_ligands, T, dtype=int)
    converged_mask = np.zeros(n_ligands, dtype=bool)

    l_idx_arr = lr_mapping.columns.get_indexer(lr_info['ligand'])
    r_idx_arr = lr_mapping.index.get_indexer(lr_info['receptor'])

    for s in (tqdm(range(1, T + 1), desc='step') if verbose else range(1, T + 1)):
        L_prev = L.copy()
        diffusion_L = -D_L * lap_mtx.dot(L)

        binding_L.fill(0)
        binding_R.fill(0)
        binding_C.fill(0)
        unbinding_L.fill(0)
        unbinding_R.fill(0)
        unbinding_C = k_off * C

        for c_idx, (l_idx, r_idx) in enumerate(zip(l_idx_arr, r_idx_arr)):
            total_competing_R = R.dot(mask_rec[l_idx])
            L_eff = L[:, l_idx] / (1 + alpha * total_competing_R)

            total_competing_L = L.dot(mask_lig[r_idx])
            R_eff = R[:, r_idx] / (1 + beta * total_competing_L)

            pair_binding = k_on * L_eff * R_eff

            binding_L[:, l_idx] += pair_binding
            binding_R[:, r_idx] += pair_binding
            binding_C[:, c_idx] += pair_binding

            pair_unbinding = unbinding_C[:, c_idx]
            unbinding_L[:, l_idx] += pair_unbinding
            unbinding_R[:, r_idx] += pair_unbinding

        dL_dt, dR_dt, dC_dt = _compute_grd_derivatives(
            model=model,
            L=L,
            R=R,
            C=C,
            diffusion_L=diffusion_L,
            binding_L=binding_L,
            binding_R=binding_R,
            binding_C=binding_C,
            unbinding_L=unbinding_L,
            unbinding_R=unbinding_R,
            unbinding_C=unbinding_C,
            S_L_feedback=base_L,
            S_R_feedback=base_R,
            lambda_L=lambda_L,
            lambda_R=lambda_R,
            delta_L=delta_L,
            delta_R=delta_R,
            delta_C=delta_C,
        )

        L = np.maximum(0, L + dt * dL_dt)
        R = np.maximum(0, R + dt * dR_dt)
        C = np.maximum(0, C + dt * dC_dt)

        current_new_ligand_mask = (L > 0) & initial_zero_mask
        if np.any(current_new_ligand_mask):
            new_ligand_indices = np.argwhere(current_new_ligand_mask)
            new_diffusion_ligand.append({
                'step': s,
                'new_indices': new_ligand_indices,
            })

        diff_vec = np.linalg.norm(L - L_prev, axis=0) / (np.linalg.norm(L_prev, axis=0) + 1e-9)
        current_converged = (diff_vec < tol) & (~converged_mask)
        if np.any(current_converged):
            convergence_steps[current_converged] = s
            converged_mask[current_converged] = True
            if verbose and np.all(converged_mask):
                print(f"All ligands converged by step {s}")

        ligand_step_list.append(L.copy())
        receptor_step_list.append(R.copy())
        complex_step_list.append(C.copy())

    convergence_df = pd.DataFrame({
        'ligand': unique_ligands,
        'convergence_step': convergence_steps,
    })

    return ligand_step_list, receptor_step_list, complex_step_list, new_diffusion_ligand, convergence_df



def pathway_activity_agg_with_plots(adata, lr_info, time_step, pathway_name, prefix='grd', p_col='pathway', res_dir=None, plt_show=False, **kwargs):
    """
    Aggregate complex activity by pathway.

    Parameters
    ----------
    adata : AnnData
        AnnData object with .obs containing per-complex activity.
    lr_info : pd.DataFrame
        DataFrame with complex index and a 'pathway' column.
    time_step : int
        Time step to aggregate.
    prefix : str
        Prefix for the column names in .obs.
    p_col : str
        Column name in lr_info indicating pathway.
    """
    if res_dir is not None:
        p_res_dir = res_dir / pathway_name
        os.makedirs(p_res_dir, exist_ok=True)
    lr_info = lr_info[lr_info[p_col] == pathway_name]
    # print(f"======  Number of lr pairs belongs to {pathway_name}: {lr_info.shape[0]}  =====")
    obs_pathway_col = f"{prefix}_{pathway_name}_activity_step{time_step}"
    
    adata = adata.copy()
    for _, row in lr_info.iterrows():
        complex_name = row['complex']
        obs_complex_col = f"{prefix}_{complex_name}_step{time_step}"

        if obs_pathway_col not in adata.obs:
            adata.obs[obs_pathway_col] = 0.0

        # print(f"Add {obs_complex_col} into {obs_pathway_col}")

        adata.obs[obs_pathway_col] += adata.obs[obs_complex_col]
    # print(f"Add {obs_pathway_col} into adata.obs")

    # if plt_show:
    #     sc.pl.spatial(adata=adata, color=f"{prefix}_{pathway_name}_activity_step{time_step}", frameon=False, **kwargs)
    # else: # could save the figure is res_dir not None
    #     if res_dir is not None:
    #         sc.pl.spatial(adata=adata, color=f"{prefix}_{pathway_name}_activity_step{time_step}", frameon=False, show=False, **kwargs)
    #         plt.savefig(p_res_dir / f"pathway_{pathway_name}_activity.png", dpi=300, bbox_inches='tight')
    #         plt.close()

    for index, row in lr_info.iterrows():
        l_name, r_name, c_name = row['ligand'], row['receptor'], row['complex']
        ligand_key = f"{prefix}_{l_name}_step{time_step}"
        receptor_key = f"{prefix}_{r_name}_step{time_step}"
        complex_key = f"{prefix}_{c_name}_step{time_step}"

        color_keys = []
        # for key in [ligand_key, receptor_key, complex_key]:
        for key in [complex_key]:
            if key in adata.obs.columns or key in adata.var_names:
                color_keys.append(key)

    # for key in color_keys:
    #     if plt_show:
    #         sc.pl.spatial(adata, color=color_keys, frameon=False, **kwargs)
    #     else:
    #         if res_dir is not None:
    #             sc.pl.spatial(adata, color=color_keys, frameon=False, show=False, **kwargs)
    #             plt.savefig(p_res_dir / f"pathway_{pathway_name}_lig_{l_name}_rec_{r_name}_activity.png", dpi=300, bbox_inches='tight')
    #             plt.close()
                
    return adata
