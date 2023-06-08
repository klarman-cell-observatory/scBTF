import os
import numpy as np
import pandas as pd
import anndata as ad

from typing import Optional, Union
from adpbulk import ADPBulk
from anndata import AnnData
from tqdm import tqdm
import pkg_resources


class SingleCellTensor:
    """
    Represent tensorized single cell data

    Parameters
    ----------
    tensor: np.ndarray
        Tensor data with shape (samples, celltypes, genes).
    gene_list: List[str]
        List of genes.
    sample_list: List[str], optional
        List of sample names, default is None.
    celltype_list: List[str], optional
        List of cell types, default is None.
    region_list: List[str], optional
        List of regions, default is None.
    normalized: bool, optional
        Boolean flag indicating if the tensor is normalized, default is False.
    sample_features: pd.DataFrame, optional
        DataFrame with sample features, default is None.

    """

    def __init__(
            self,
            tensor: np.ndarray,
            gene_list: list[str],
            sample_list: Optional[list[str]] = None,
            celltype_list: Optional[list[str]] = None,
            region_list: Optional[list[str]] = None,
            normalized: bool = False,
            sample_features: Optional[pd.DataFrame] = None
    ) -> None:

        self.gene_list = gene_list
        self.sample_list = sample_list
        self.celltype_list = celltype_list
        self.region_list = region_list

        self.tensor = tensor
        self.normalized = normalized
        self.sample_features = sample_features


    @staticmethod
    def from_anndata(
            adata: ad.AnnData,
            sample_label: str,
            celltype_label: str,
            normalize: bool = True,
            variance_scale: bool = False,
            sqrt_transform: bool = False,
            scale_to: int = 1e6,
            cell_types: Optional[list[str]] = None,
            enrich_db_genes_only: bool = False,
            hgnc_approved_genes_only: bool = False,
            custom_gene_set: Optional[list[str]] = None,
            filter_gene_count: int = 10,
            enrich_dbs: list[str] = ['GO_Biological_Process_2021']  # ,'KEGG_2021_Human', 'Reactome_2022']
    ) -> 'SingleCellTensor':
        """

        Compose tensor from single cell anndata.

        Parameters
        ----------
        adata: ad.AnnData
            Single cell Anndata object.
        sample_label: str
            Sample label.
        celltype_label: str
            Cell type label.
        normalize: bool, optional
            Boolean flag indicating if the tensor should be normalized, default is True.
        variance_scale: bool, optional
            Boolean flag indicating if variance scaling should be applied to the tensor, default is False.
        sqrt_transform: bool, optional
            Boolean flag indicating if square root transform should be applied to the tensor, default is False.
        scale_to: int, optional
            Value to scale tensor to, default is 1e6.
        cell_types: List[str], optional
            List of cell types, default is None.
        enrich_db_genes_only: bool, optional
            Boolean flag indicating if only enriched genes should be included, default is False.
        hgnc_approved_genes_only: bool, optional
            Boolean flag indicating if only HGNC approved genes should be included, default is False.
        custom_gene_set: List[str], optional
            List of genes to include, default is None.
        filter_gene_count: int, optional
            Minimum gene count for filtering, default is 10.
        enrich_dbs: List[str], optional
            List of enrichment databases to use, default is ['GO_Biological_Process_2021'].

        Returns
        -------
        SingleCellTensor
            Tensor object.

        """

        adata = adata.raw.to_adata()

        adata = SingleCellTensor.filter_adata(
            adata, cell_types, celltype_label, custom_gene_set, enrich_db_genes_only,
            enrich_dbs, filter_gene_count, hgnc_approved_genes_only
        )

        adpb = ADPBulk(adata, [sample_label, celltype_label])
        pseudobulk_matrix = adpb.fit_transform()
        sample_meta = adpb.get_meta()

        if normalize:
            row_sums = pseudobulk_matrix.sum(axis=1)
            pseudobulk_matrix = pseudobulk_matrix / row_sums[:, np.newaxis] * scale_to

        if variance_scale:
            std = pseudobulk_matrix.std()
            std[std == 0] = 1
            pseudobulk_matrix /= std

        if sqrt_transform:
            pseudobulk_matrix = pseudobulk_matrix.apply(np.sqrt)

        tensor_sample_list = sample_meta[sample_label].unique()
        tensor_celltype_list = sample_meta[celltype_label].unique()
        tensor_gene_list = pseudobulk_matrix.columns.tolist()

        sample_meta_dict = {(row[sample_label], row[celltype_label]): row['SampleName'] for idx, row in
                            sample_meta.iterrows()}

        tensor = np.zeros((len(tensor_sample_list), len(tensor_celltype_list), len(tensor_gene_list)))

        for i in tqdm(range(len(tensor_sample_list)), desc=f"Building tensor from matrix"):
            for j in range(len(tensor_celltype_list)):
                key = (tensor_sample_list[i], tensor_celltype_list[j])
                if key in sample_meta_dict:
                    tensor[i, j, :] = pseudobulk_matrix.loc[sample_meta_dict[key], tensor_gene_list]

        sample_features = SingleCellTensor.adata_obs_to_summary_df(adata, sample_label=sample_label)

        return SingleCellTensor(tensor,
                                celltype_list=tensor_celltype_list,
                                gene_list=tensor_gene_list,
                                sample_list=tensor_sample_list,
                                normalized=normalize,
                                sample_features=sample_features
                                )

    @staticmethod
    def filter_adata(
        adata: AnnData,
        cell_types: Optional[list] = None,
        celltype_label: Optional[str] = 'celltype',
        custom_gene_set: Optional[list] = None,
        enrich_db_genes_only: bool = False,
        enrich_dbs: Optional[list] = None,
        filter_gene_count: Optional[int] = None,
        hgnc_approved_genes_only: bool = False
    ) -> AnnData:
        """
        Filter AnnData object according to specified criteria.

        Parameters
        ----------
        adata : AnnData
            An annotated data matrix.
        cell_types : list or None, optional
            List of cell types to retain, by default None.
        celltype_label : str, optional
            Name of the column in `adata.obs` where cell types are stored, by default 'celltype'.
        custom_gene_set : list or None, optional
            List of genes to retain, by default None.
        enrich_db_genes_only : bool, optional
            Whether to keep only genes that are present in at least one of the enrich databases, by default False.
        enrich_dbs : list or None, optional
            List of enrich databases to use, by default None.
        filter_gene_count : int or None, optional
            Minimum total number of UMI counts per gene to keep the gene, by default None.
        hgnc_approved_genes_only : bool, optional
            Whether to keep only HGNC-approved protein-coding genes, by default False.

        Returns
        -------
        AnnData
            The filtered AnnData object.
        """

        if cell_types is not None:
            adata = adata[adata.obs[celltype_label].isin(cell_types)].copy()

        if enrich_db_genes_only:
            go_genes = set.union(
                *[set(g for gset in SingleCellTensor.parse_gmt(db).values() for g in gset) for db in enrich_dbs])
            adata = adata[:, adata.var_names.isin(go_genes)].copy()

        if hgnc_approved_genes_only:
            hgnc_path = pkg_resources.resource_filename("scBTF", "resources/gene_sets/hgnc_complete_set_2020-07-01.txt")
            hgnc_full = pd.read_csv(hgnc_path, sep='\t', low_memory=False)
            selected_genes = hgnc_full[
                (hgnc_full.status == 'Approved') & (hgnc_full.locus_group == 'protein-coding gene')]
            adata = adata[:, adata.var_names.str.split('.').str[0].isin(selected_genes['symbol'])].copy()

        if custom_gene_set is not None:
            adata = adata[:, adata.var_names.isin(custom_gene_set)].copy()

        if filter_gene_count is not None:
            adata = adata[:, adata.X.sum(axis=0) > filter_gene_count].copy()

        return adata

    @staticmethod
    def from_anndata_with_regions(
            adata: ad.AnnData,
            sample_label: str,
            celltype_label: str,
            region_label: str,
            normalize: bool = True,
            variance_scale: bool = False,
            sqrt_transform: bool = False,
            scale_to: int = 1e6,
            cell_types: list[str] = None,
            enrich_db_genes_only: bool = False,
            hgnc_approved_genes_only: bool = False,
            custom_gene_set: list[str] = None,
            filter_gene_count: int = 10,
            enrich_dbs: list[str] = ['GO_Biological_Process_2021']  # ,'KEGG_2021_Human', 'Reactome_2022']
    ):
        """
        Compose tensor from single cell anndata

        Returns
        -------
        tensors
            dictionary with two entries raw and normalized
            each is a sample x celltype x gene tensor
        """

        adata = adata.raw.to_adata()

        adata = SingleCellTensor.filter_adata(
            adata, cell_types, celltype_label, custom_gene_set, enrich_db_genes_only,
            enrich_dbs, filter_gene_count, hgnc_approved_genes_only
        )

        adpb = ADPBulk(adata, [sample_label, celltype_label, region_label])
        pseudobulk_matrix = adpb.fit_transform()
        sample_meta = adpb.get_meta()

        if normalize:
            row_sums = pseudobulk_matrix.sum(axis=1)
            pseudobulk_matrix = pseudobulk_matrix / row_sums[:, np.newaxis] * scale_to

        if variance_scale:
            std = pseudobulk_matrix.std()
            std[std == 0] = 1
            pseudobulk_matrix /= std

        if sqrt_transform:
            pseudobulk_matrix = pseudobulk_matrix.apply(np.sqrt)

        tensor_sample_list = sample_meta[sample_label].unique()
        tensor_celltype_list = sample_meta[celltype_label].unique()
        tensor_region_list = sample_meta[region_label].unique()
        tensor_gene_list = pseudobulk_matrix.columns.tolist()

        sample_meta_dict = {(row[sample_label], row[celltype_label], row[region_label]): row['SampleName']
                            for idx, row in sample_meta.iterrows()}

        tensor = np.zeros(
            (len(tensor_sample_list), len(tensor_celltype_list), len(tensor_region_list), len(tensor_gene_list)))
        for i in tqdm(range(len(tensor_sample_list)), desc=f"Building tensor from pseudobulk matrix"):
            for j in range(len(tensor_celltype_list)):
                for k in range(len(tensor_region_list)):
                    key = (tensor_sample_list[i], tensor_celltype_list[j], tensor_region_list[k])
                    if key in sample_meta_dict:
                        tensor[i, j, k, :] = pseudobulk_matrix.loc[sample_meta_dict[key], tensor_gene_list]

        sample_features = SingleCellTensor.adata_obs_to_summary_df(adata, sample_label=sample_label)

        return SingleCellTensor(tensor,
                                celltype_list=tensor_celltype_list,
                                gene_list=tensor_gene_list,
                                sample_list=tensor_sample_list,
                                region_list=tensor_region_list,
                                normalized=normalize,
                                sample_features=sample_features
                                )

    @staticmethod
    def from_anndata_ligand_receptor(
            adata: ad.AnnData,
            sample_label: str,
            celltype_label: str,
            normalize: bool = True,
            variance_scale: bool = False,
            sqrt_transform: bool = False,
            scale_to: int = 1e6,
            cell_types: list[str] = None,
            enrich_db_genes_only: bool = False,
            custom_gene_set: list[str] = None,
            filter_gene_count: int = 10,
            enrich_dbs: list[str] = ['GO_Biological_Process_2021'],  # ,'KEGG_2021_Human', 'Reactome_2022']
            lr_table_path: str = None,
            communication_score: str = 'expression_mean'
    ):
        """
        Compose tensor from single cell anndata

        Returns
        -------
        tensors
            dictionary with two entries raw and normalized
            each is a sample x celltype x gene tensor
        """
        sc_tensor = SingleCellTensor.from_anndata(
            adata=adata,
            sample_label=sample_label,
            celltype_label=celltype_label,
            normalize=normalize,
            variance_scale=variance_scale,
            sqrt_transform=sqrt_transform,
            scale_to=scale_to,
            cell_types=cell_types,
            enrich_db_genes_only=enrich_db_genes_only,
            custom_gene_set=custom_gene_set,
            filter_gene_count=filter_gene_count,
            enrich_dbs=enrich_dbs
        )

        lr_table_path = lr_table_path if lr_table_path is not None else pkg_resources.resource_filename("scBTF", "resources/liana_consensus_LR.csv")
        lr_table = pd.read_csv(lr_table_path)
        lr_table = list(set([(v['source_genesymbol'], v['target_genesymbol']) for v in lr_table.T.to_dict().values()]))
        lr_table = [lr for lr in lr_table if lr[0] in sc_tensor.gene_list and lr[1] in sc_tensor.gene_list]

        tensor = np.zeros(
            (len(sc_tensor.sample_list), len(sc_tensor.celltype_list), len(sc_tensor.celltype_list), len(lr_table)))

        for d in tqdm(range(len(sc_tensor.sample_list))):
            for lr in range(len(lr_table)):
                l = sc_tensor.tensor[d, :, sc_tensor.gene_list.index(lr_table[lr][0])]
                r = sc_tensor.tensor[d, :, sc_tensor.gene_list.index(lr_table[lr][1])]
                tensor[d, :, :, lr] = SingleCellTensor.compute_ccc_matrix(l, r, communication_score)

        sc_tensor.tensor = tensor
        sc_tensor.region_list = sc_tensor.celltype_list
        sc_tensor.gene_list = [lr[0] + '_' + lr[1] for lr in lr_table]

        return sc_tensor

    @staticmethod
    def parse_gmt(gmt):
        """
        Retrieve genes that make up gene set enrichment libraries
        """
        BASE_ENRICHR_URL = "http://maayanlab.cloud"
        DEFAULT_CACHE_PATH = "resources/gene_sets"
        os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)

        def _download_libraries(libname: str):
            import requests
            ENRICHR_URL = BASE_ENRICHR_URL + "/Enrichr/geneSetLibrary"
            query_string = "?mode=text&libraryName=%s"
            response = requests.get(ENRICHR_URL + query_string % libname, timeout=None)
            if not response.ok:
                raise Exception("Error fetching gene set library, input name is correct for the organism you've set?.")
            genesets_dict = {}
            outname = "Enrichr.%s.gmt" % libname  # pattern: database.library.gmt
            gmtout = open(os.path.join(DEFAULT_CACHE_PATH, outname), "w")
            for line in response.iter_lines(chunk_size=1024, decode_unicode="utf-8"):
                line = line.strip().split("\t")
                k = line[0]
                v = map(lambda x: x.split(",")[0], line[2:])
                v = list(filter(lambda x: True if len(x) else False, v))
                genesets_dict[k] = v
                outline = "%s\t%s\t%s\n" % (k, line[1], "\t".join(v))
                gmtout.write(outline)
            gmtout.close()

            return genesets_dict

        tmpname = "Enrichr." + gmt + ".gmt"
        tempath = os.path.join(DEFAULT_CACHE_PATH, tmpname)
        if os.path.isfile(tempath):
            with open(tempath) as genesets:
                genesets_dict = {
                    line.strip().split("\t")[0]: line.strip().split("\t")[2:]
                    for line in genesets.readlines()
                }
            return genesets_dict
        else:
            return _download_libraries(gmt)

    @staticmethod
    def adata_obs_to_summary_df(adata, sample_label):
        """
        Convert adata obs into a summary dataframe with one row for each unique `sample_label`
        """
        idx = [np.where(adata.obs[sample_label] == sample)[0][0] for sample in
               adata.obs[sample_label].value_counts().index]

        summary_df = adata.obs.iloc[idx, :].sort_values(by=sample_label)
        summary_df.index = summary_df[sample_label]

        return summary_df

    @staticmethod
    def compute_ccc_matrix(prot_a_exp, prot_b_exp, communication_score='expression_product'):
        """
        adapted from Liana https://github.com/saezlab/liana
        """
        if communication_score == 'expression_product':
            communication_scores = np.outer(prot_a_exp, prot_b_exp)
        elif communication_score == 'expression_mean':
            communication_scores = (np.outer(prot_a_exp, np.ones(prot_b_exp.shape)) + np.outer(
                np.ones(prot_a_exp.shape), prot_b_exp)) / 2.
        elif communication_score == 'expression_gmean':
            communication_scores = np.sqrt(np.outer(prot_a_exp, prot_b_exp))
        else:
            raise ValueError("Not a valid communication_score")
        return communication_scores
