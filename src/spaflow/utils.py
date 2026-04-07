import pandas as pd
import numpy as np
import scipy.sparse as ss
import squidpy as sq
import scanpy as sc

def create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)),
                               shape=(diag.size, diag.size))
    return sparse_mtx

def get_neighbors(adata,
                      spatial_key='spatial',
                      coord_type='generic',
                      weighted=False,
                      **squidpy_kwargs):
    adata = adata.copy()
    sq.gr.spatial_neighbors(
        adata,
        spatial_key=spatial_key,
        coord_type=coord_type,
        **squidpy_kwargs
    )

    # Add weight to the edges or not
    if not weighted:
        adj_mtx = adata.obsp['spatial_connectivities']
    else:
        adj_mtx = adata.obsp['spatial_distances']
        adj_mtx.data = 1.0 / adj_mtx.data

    deg_arr = np.array(adj_mtx.sum(axis=1)).flatten()

    return adata, adj_mtx, deg_arr

def get_laplacian_mtx(adata,
                      spatial_key='spatial',
                      coord_type='generic',
                      normalization=True,
                      weighted=False,
                      **squidpy_kwargs):
    """
    Calculate spatial adjacency matrix and Laplacian matrix using squidpy.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    spatial_key : str
        Key in adata.obsm containing spatial coordinates.
    coord_type : str
        Coordinate type ("grid" or "generic").
    normalization : bool
        Whether to calculate normalized Laplacian matrix.
    weighted : bool
        Whether to calculate weighted Laplacian matrix.
    squidpy_kwargs : dict
        Additional parameters passed to squidpy.gr.spatial_neighbors.

    Returns
    -------
    lap_mtx : scipy.sparse matrix
        Laplacian matrix.
    adj_mtx : scipy.sparse matrix
        Adjacency matrix.

    Examples
    --------
    Stereo-seq data example call:
    >>> lap_mtx, adj_mtx = get_laplacian_mtx(
    ...     adata,
    ...     spatial_key='spatial',
    ...     coord_type='generic',
    ...     normalization=False,
    ...     weighted=True,
    ...     delaunay=True,
    ...     radius=50
    ... )

    Visium data example call (default grid structure):
    >>> lap_mtx, adj_mtx = get_laplacian_mtx(
    ...     adata,
    ...     spatial_key='spatial',
    ...     coord_type='grid',
    ...     normalization=True,
    ...     weighted=False
    ... )
    """
    init_adata, init_adj_mtx, init_deg_arr = get_neighbors(adata,
                      spatial_key=spatial_key,
                      coord_type=coord_type,
                      weighted=weighted,
                      **squidpy_kwargs)

    adj_mtx = None
    deg = None

    # zero_degree_nodes = np.where(init_deg_arr == 0)[0]
    # if len(zero_degree_nodes) > 0:
    #     print(f"Find {len(zero_degree_nodes)} cells with degree = 0")
    #     keep_nodes_idx = np.where(init_deg_arr > 0)[0]
    #     keep_nodes_names = adata.obs_names[keep_nodes_idx]

    #     filter_adata = adata[keep_nodes_names, :].copy()
    #     filter_adata, filter_adj_mtx, filter_deg_arr = get_neighbors(filter_adata,
    #                   spatial_key=spatial_key,
    #                   coord_type=coord_type,
    #                   weighted=weighted,
    #                   **squidpy_kwargs)
    #     adj_mtx = filter_adj_mtx
    #     deg = filter_deg_arr
    #     out_adata = filter_adata
    #     print(f"Cell number after removing: {len(out_adata.obs_names)}")
    # else:
    adj_mtx = init_adj_mtx
    deg = init_deg_arr
    out_adata = init_adata

    deg_mtx = ss.diags(deg.flatten())

    # Check normalization
    if normalization:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt_mtx = create_degree_mtx(deg_inv_sqrt)
        lap_mtx = ss.identity(deg_mtx.shape[0]) - deg_inv_sqrt_mtx @ adj_mtx @ deg_inv_sqrt_mtx
    else:
        # Un-normalized: L = D - A
        lap_mtx = deg_mtx - adj_mtx

    out_adata.obsp['laplacian_matrix'] = lap_mtx
    out_adata.obsp['adjacency_matrix'] = adj_mtx
    return out_adata


def concat_lr_adata(adata, ligand_step_list, receptor_step_list, complex_step_list, lr_info, 
                    prefix='grd', last_only=True, keep_gene=True):    
    unique_ligands = pd.unique(lr_info['ligand']).tolist()
    unique_receptors = pd.unique(lr_info['receptor']).tolist()

    if last_only:
        last_step_idx = len(ligand_step_list) - 1
        
        columns = [f'{prefix}_{l}_step{last_step_idx}' for l in unique_ligands]
        ligand_df_cat = pd.DataFrame(ligand_step_list[last_step_idx], index=adata.obs_names, columns=columns)
        
        columns = [f'{prefix}_{r}_step{last_step_idx}' for r in unique_receptors]
        receptor_df_cat = pd.DataFrame(receptor_step_list[last_step_idx], index=adata.obs_names, columns=columns)
        
        columns = [f'{prefix}_{c}_step{last_step_idx}' for c in lr_info['complex']]
        complex_df_cat = pd.DataFrame(complex_step_list[last_step_idx], index=adata.obs_names, columns=columns)
        
    else:
        ligand_df_list = []
        for i, ligand in enumerate(ligand_step_list):
            columns = [f'{prefix}_{l}_step{i}' for l in unique_ligands]
            ligand_df = pd.DataFrame(ligand, index=adata.obs_names, columns=columns)
            ligand_df_list.append(ligand_df)
        ligand_df_cat = pd.concat(ligand_df_list, axis=1)

        receptor_df_list = []
        for i, receptor in enumerate(receptor_step_list):
            columns = [f'{prefix}_{r}_step{i}' for r in unique_receptors]
            receptor_df = pd.DataFrame(receptor, index=adata.obs_names, columns=columns)
            receptor_df_list.append(receptor_df)
        receptor_df_cat = pd.concat(receptor_df_list, axis=1)
        
        complex_df_list = []
        for i, complex in enumerate(complex_step_list):
            columns = [f'{prefix}_{c}_step{i}' for c in lr_info['complex']]
            complex_df = pd.DataFrame(complex, index=adata.obs_names, columns=columns)
            complex_df_list.append(complex_df)
        complex_df_cat = pd.concat(complex_df_list, axis=1)
    
    
    combined_df = pd.concat((ligand_df_cat, receptor_df_cat, complex_df_cat), axis=1)

    if not keep_gene:
        # new grd adata, only with ligand/receptor/complex concentration
        grd_adata = sc.AnnData(X=combined_df.values, obs=adata.obs.copy(), var=pd.DataFrame(index=combined_df.columns))
    
        
        grd_adata.uns = adata.uns.copy()
        grd_adata.obsm = adata.obsm.copy()
        grd_adata.obsp = adata.obsp.copy()

    else:
        grd_adata = adata.copy()
        # grd_adata.obsm['spagrd_score'] = combined_df
    grd_adata.obs = pd.concat([grd_adata.obs, combined_df], axis=1)
    return grd_adata
