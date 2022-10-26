import os
import numpy as np
import anndata as ad

from typing import Optional
from adpbulk import ADPBulk
from tqdm import tqdm


class SingleCellTensor:
    """
    Represent tensorized single cell data

    """

    def __init__(
            self,
            tensor,
            gene_list: list[str],
            sample_list: Optional[list[str]] = None,
            celltype_list: Optional[list[str]] = None,
            region_list: Optional[list[str]] = None,
            normalized: bool = False,
            sample_features=None

    ):

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
            normalize: bool = False,
            cell_types: list[str] = None,
            enrich_db_genes_only: bool = False,
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

        if cell_types is not None:
            adata = adata[adata.obs[celltype_label].isin(cell_types)].copy()

        if enrich_db_genes_only:
            go_genes = set.union(
                *[set(g for gset in SingleCellTensor.parse_gmt(db).values() for g in gset) for db in enrich_dbs])
            adata = adata[:, adata.var_names.isin(go_genes)].copy()

        adpb = ADPBulk(adata, [sample_label, celltype_label])
        pseudobulk_matrix = adpb.fit_transform()
        sample_meta = adpb.get_meta()

        if normalize:
            row_sums = pseudobulk_matrix.sum(axis=1)
            pseudobulk_matrix = pseudobulk_matrix / row_sums[:, np.newaxis] * 1e6

        tensor_sample_list = sample_meta[sample_label].unique()
        tensor_celltype_list = [ctype for ctype in sample_meta[celltype_label].unique()]
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
    def parse_gmt(gmt):
        """
        Retrieve genes that make up gene set enrichment libraries 
        """
        BASE_ENRICHR_URL = "http://maayanlab.cloud"
        DEFAULT_CACHE_PATH = "data/gene_sets"
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
