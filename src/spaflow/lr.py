import pandas as pd
import scipy.sparse as ss
import anndata
import numpy as np
import os
import networkx as nx
from collections import defaultdict

# deprecated
def integrate_pathways(df_ligrec, l_name='ligand', r_name = 'receptor', p_name = 'pathway'):
    """
    Intergrated pathways if ligand/receptor overlap.
    """
    df_ligrec_expanded = df_ligrec.copy()

    df_ligrec_expanded[f'unique_{l_name}'] = df_ligrec_expanded[l_name].str.split('_')
    df_ligrec_expanded = df_ligrec_expanded.explode(f'unique_{l_name}')

    df_ligrec_expanded[f'unique_{r_name}'] = df_ligrec_expanded[r_name].str.split('_')
    df_ligrec_expanded = df_ligrec_expanded.explode(f'unique_{r_name}')

    df_ligrec_expanded = df_ligrec_expanded.reset_index(drop=True)

    pathway_to_entities = defaultdict(set)
    for _, row in df_ligrec_expanded.iterrows():
        pathway = row[p_name]
        ligand = row[f'unique_{l_name}']
        receptor = row[f'unique_{r_name}']
        pathway_to_entities[pathway].update([ligand, receptor])

    G = nx.Graph()
    pathways = list(pathway_to_entities.keys())
    for i in range(len(pathways)):
        for j in range(i + 1, len(pathways)):
            pi, pj = pathways[i], pathways[j]
            if pathway_to_entities[pi] & pathway_to_entities[pj]:
                G.add_edge(pi, pj)

    component_map = {}
    for component in nx.connected_components(G):
        component_pathway = '_'.join(sorted(component))
        for p in component:
            component_map[p] = component_pathway

    for pathway in pathway_to_entities:
        if pathway not in component_map:
            component_map[pathway] = pathway

    df_ligrec_expanded['integrated_pathway'] = df_ligrec_expanded[p_name].map(component_map)
    df_ligrec = df_ligrec_expanded[[l_name, r_name, p_name, 'integrated_pathway', 'signaling type']]
    df_ligrec = df_ligrec.drop_duplicates()

    df_ligrec['complex'] = df_ligrec[l_name] + "-" + df_ligrec[r_name]
    
    return df_ligrec


# NOTE: Adapted from https://github.com/zcang/COMMOT/blob/5dc3be94d7856d0736410d98be67003dbc3fe649/commot/preprocessing/_ligand_receptor_database.py#L58
def filter_lr_database(
    adata: anndata.AnnData,
    species = 'human',
    database = "CellChat",
    signaling_type = "Secreted Signaling",
    heteromeric_delimiter: str = "_",
    heteromeric_rule: str = "min",
    filter_criteria: str = "min_cell_pct",
    min_cell: int = 100,
    min_cell_pct: float = 0.05
):
    """
    Filter ligand-receptor pairs.

    Parameters
    ----------
    adata
        The AnnData object of gene expression. Unscaled data (minimum being zero) is expected.
    heteromeric_delimiter
        If heteromeric notations are used for ligands and receptors, the character separating the heteromeric units.
    heteromeric_rule
        When  heteromeric is True, the rule to quantify the level of a heteromeric ligand or receptor. Choose from minimum ('min') and average ('ave').
    filter_criteria
        Use either cell percentage ('min_cell_pct') or cell numbers (min_cell) to filter genes.
    min_cell
        If filter_criteria is 'min_cell', the LR-pairs with both ligand and receptor detected in greater than or equal to min_cell cells are kept.
    min_cell_pct
        If filter_criteria is 'min_cell_pct', the LR-pairs with both ligand and receptor detected in greater than or equal to min_cell_pct percentage of cells are kepts.

    Returns
    -------
    df_ligrec_filtered: pd.DataFrame
        A pandas DataFrame of the filtered ligand-receptor pairs.

    """
    lr_base_path = os.path.join(os.path.dirname(__file__), "LRdatabase")

    if database == 'CellChat':
        df_ligrec = pd.read_csv(os.path.join(lr_base_path, f"CellChat/CellChatDB.ligrec.{species}.csv"), index_col=0)
        df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
    elif database == 'CellPhoneDB_v4.0':
        df_ligrec = pd.read_csv(os.path.join(lr_base_path, f"CellPhoneDB_v4.0/CellPhoneDBv4.0.{species}.csv"), index_col=0)
        df_ligrec['2'] = [f"p{i+1}" for i in range(len(df_ligrec))]
        df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
         
    ncell = adata.shape[0]
    all_genes = list(adata.var_names)
    gene_ncell = np.array( (adata.X > 0).sum(axis=0) ).reshape(-1)
    ligrec_list = []
    genes_keep = []

    tmp_genes = list(set(df_ligrec.iloc[:,0]).union(set(df_ligrec.iloc[:,1])))
    for het_gene in tmp_genes:
        genes = het_gene.split(heteromeric_delimiter)
        gene_found = True
        for gene in genes:
            if not gene in all_genes:
                gene_found = False
        if not gene_found: continue
        keep = True
        if filter_criteria == 'min_cell_pct' and heteromeric_rule == 'min':
            for gene in genes:
                if gene_ncell[all_genes.index(gene)] / ncell < min_cell_pct:
                    keep = False
        elif filter_criteria == 'min_cell' and heteromeric_rule == 'min':
            for gene in genes:
                if gene_ncell[all_genes.index(gene)] < min_cell:
                    keep = False
        elif heteromeric_rule == 'ave':
            ave_ncell = []
            for gene in genes:
                ave_ncell.append( gene_ncell[all_genes.index(gene)] )
            if filter_criteria == 'min_cell_pct':
                if np.mean(ave_ncell) / ncell < min_cell_pct:
                    keep = False
            elif filter_criteria == 'min_cell':
                if np.mean(ave_ncell) < min_cell:
                    keep = False
        if keep:
            genes_keep.append(het_gene)
    for i in range(df_ligrec.shape[0]):
        if df_ligrec.iloc[i,0] in genes_keep and df_ligrec.iloc[i,1] in genes_keep:
            ligrec_list.append(list(df_ligrec.iloc[i,:]))
    
    df_ligrec = pd.DataFrame(data=ligrec_list)
    df_ligrec = df_ligrec.rename(columns={0: 'ligand', 1: 'receptor', 2: 'pathway', 3: 'signaling type'})

    df_ligrec = integrate_pathways(df_ligrec)



    return df_ligrec


