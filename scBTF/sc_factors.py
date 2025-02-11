import pickle
import os
import scipy
import torch
import rich
import concurrent.futures
import fastcluster

import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu

from .sc_tensor import SingleCellTensor
from .bayesian_parafac import BayesianCP


class Factorization:
    """ Single group of factors

    Parameters
    ----------
    sample_factor : dict
        A dictionary representing sample factor
    celltype_factor : dict
        A dictionary representing celltype factor
    region_factor : dict, optional
        A dictionary representing region factor (default is None)
    gene_factor : dict, optional
        A dictionary representing gene factor (default is None)
    """

    def __init__(self, sample_factor, celltype_factor, region_factor=None, gene_factor=None):
        self.sample_factor = sample_factor
        self.celltype_factor = celltype_factor
        self.gene_factor = gene_factor
        self.region_factor = region_factor

        self.has_region = region_factor is not None

    def get_means(self):
        """
        Returns the mean values of the factors.

        Returns
        -------
        list
            A list of mean values for sample, celltype, region (if available) and gene factor
        """

        if self.has_region:
            return [
                self.sample_factor['mean'],
                self.celltype_factor['mean'],
                self.region_factor['mean'],
                self.gene_factor['mean']
            ]
        else:
            return [
                self.sample_factor['mean'],
                self.celltype_factor['mean'],
                self.gene_factor['mean']
            ]



class FactorizationSet:
    """
    A collection of factorizations indexed by rank and restart number

    Parameters
    ----------
    sc_tensor : SingleCellTensor, optional
        SingleCellTensor object that stores the input data, by default None.

    Attributes
    ----------
    sc_tensor : SingleCellTensor
        SingleCellTensor object that stores the input data.
    factorizations : defaultdict(dict)
        A dictionary containing factorization objects indexed by rank and restart number.
    factorization_parameters : defaultdict(dict)
        A dictionary containing factorization parameters indexed by rank and restart number.
    all_cluster_metrics : None
        Placeholder for storing all cluster metrics.
    all_gene_consensus_matrix : None
        Placeholder for storing all gene consensus matrix.
    gene_consensus_lax : None
        Placeholder for storing gene consensus matrix using a relaxed threshold.
    """

    def __init__(self, sc_tensor: SingleCellTensor = None):

        self.sc_tensor = sc_tensor
        self.factorizations = defaultdict(dict)
        self.factorization_parameters = defaultdict(dict)

        self.all_cluster_metrics = None
        self.all_gene_consensus_matrix = None
        self.gene_consensus_lax = None

    def add_factorization(self, rank: int, restart_index: int, factorization: Factorization):
        """
        Add a factorization object to the factorizations dictionary.

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The restart index of the factorization.
        factorization : Factorization
            The factorization object to be added.
        """
        self.factorizations[rank][restart_index] = factorization

    def get_factorization(self, rank: int, restart_index: int):
        """
        Retrieve a factorization object from the factorizations dictionary.

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The restart index of the factorization.

        Returns
        -------
        Factorization
            The factorization object.
        """
        return self.factorizations[rank][restart_index]

    def get_ranks(self):
        """
        Retrieve a list of ranks.

        Returns
        -------
        KeysView
            A view object containing the keys of the factorizations dictionary.
        """
        return self.factorizations.keys()

    def add_mean_factorization(self, rank: int, restart_index: int, means: list):
        """
        Add a mean factorization object to the factorizations dictionary.
        If the length of the means is 3, the factorization corresponds to a 3-dimensional tensor of shape
        sample x celltype x gene.

        If the length of the means is 4, the factorization corresponds to a 4-dimensional tensor of shape
        sample x celltype x region x gene.

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The restart index of the factorization.
        means : list
            A list containing sample, cell type, region (optional), and gene factors.

        Raises
        ------
        Exception
            If the number of dimensions in the tensor is greater than 4.

        """
        if len(means) == 3:
            self.add_factorization(
                rank, restart_index,
                Factorization(
                    sample_factor=means[0],
                    celltype_factor=means[1],
                    gene_factor=means[2]
                )
            )
        elif len(means) == 4:
            self.add_factorization(
                rank, restart_index,
                Factorization(
                    sample_factor=means[0],
                    celltype_factor=means[1],
                    region_factor=means[2],
                    gene_factor=means[3]
                )
            )
        else:
            raise Exception("Implement >4d factorization")

    def add_precis_factorization(self, rank: int, restart_index: int, precis: dict):
        """
        Adds a factorization with the given rank, restart index, and dictionary of factors.

        If the length of the dictionary is 3, the factorization corresponds to a 3-dimensional tensor of shape
        sample x celltype x gene.

        If the length of the dictionary is 4, the factorization corresponds to a 4-dimensional tensor of shape
        sample x celltype x region x gene.

        Parameters
        ----------
        rank : int
            The rank of the tensor factorization.
        restart_index : int
            The restart index of the tensor factorization.
        precis : dict
            A dictionary of factors, where the keys are strings with the format 'factor_<index>',
            and the values are arrays representing the corresponding factor.

        Raises
        ------
        Exception
            If the number of dimensions in the tensor is greater than 4.

        """
        if len(precis.keys()) == 3:
            self.add_factorization(
                rank, restart_index,
                Factorization(
                    sample_factor=precis['factor_0'],
                    celltype_factor=precis['factor_1'],
                    gene_factor=precis['factor_2']
                )
            )
        elif len(precis.keys()) == 4:
            self.add_factorization(
                rank, restart_index,
                Factorization(
                    sample_factor=precis['factor_0'],
                    celltype_factor=precis['factor_1'],
                    region_factor=precis['factor_2'],
                    gene_factor=precis['factor_3']
                )
            )
        else:
            raise Exception("Implement >4d factorization")

    def add_factorization_params(self, rank: int, restart_index: int, params: dict):
        """
        Adds factorization parameters with the given rank, restart index, and dictionary of parameters.

        Parameters
        ----------
        rank : int
            The rank of the tensor factorization.
        restart_index : int
            The restart index of the tensor factorization.
        params : dict
            A dictionary of factorization parameters.

        """
        self.factorization_parameters[rank][restart_index] = params

    def variance_explained(self, rank: int, restart_index: int):
        """
        Calculates the variance explained by the tensor factorization with the given rank and restart index.

        variance_explained = 1 - (t - t')^2 / || t ||^2

        Parameters
        ----------
        rank : int
         The rank of the tensor factorization.
        restart_index : int
         The restart index of the tensor factorization.

        Returns
        -------
        float
         The variance explained by the tensor factorization with the given rank and restart index.

        """
        factorization = self.get_factorization(rank, restart_index)
        tensor_means = torch.einsum(BayesianCP.get_einsum_formula(len(self.sc_tensor.tensor.shape)),
                                    *factorization.get_means())
        tensor = torch.from_numpy(self.sc_tensor.tensor)
        return 1 - (torch.norm(tensor.float() - tensor_means.float()) ** 2 / torch.norm(tensor.float()) ** 2)

    def variance_explained_elbow_plot(self, ax=None):
        """
        Generates an elbow plot of the variance explained by the matrix factorizations.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the elbow plot. If `None`, a new figure and axes are created.

        Returns
        -------
        None
            The elbow plot is shown on the specified or newly created axes.
        """
        var_explained = pd.DataFrame({
            rank: [self.variance_explained(rank, i).item()
                   for i in range(len(self.factorizations[rank].keys()))]
            for rank in self.get_ranks()
        })
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))
        var_explained = var_explained.melt(var_name='Rank', value_name='Explained Variance')
        sns.pointplot(var_explained, x='Rank', y='Explained Variance', color='g', ax=ax)
        ax.set_ylim(min(var_explained['Explained Variance']) - 0.05, 1)
        plt.show()

    def consensus_matrix(self, rank, n_clusters=4):
        """
        Computes the consensus matrix for the specified rank and number of clusters.
        Indicates stability across multiple factorization iterations

        Parameters
        ----------
        rank : int
            The rank of the matrix factorization to use.
        n_clusters : int, optional
            The number of clusters to use when computing the consensus matrix. The default value is 4.

        Returns
        -------
        ndarray
            The consensus matrix.
        """
        dim1 = self.get_factorization(rank, 0).sample_factor['mean'].shape[0]
        cons = np.mat(np.zeros((dim1, dim1)))
        for i in self.factorizations[rank].keys():
            sample_factor = self.get_factorization(rank, i).sample_factor['mean'].numpy()
            sample_factor = sample_factor / sample_factor.sum(axis=0)
            cons += FactorizationSet.connectivity(sample_factor=sample_factor, n_clusters=n_clusters)
        return np.divide(cons, len(self.factorizations[rank].keys()))

    def cophenetic_correlation_elbow_plot(self, ax=None, n_clusters: int = 4):
        """
        Plots the elbow plot of the Cophenetic Correlation.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot or None, optional
            The subplot to plot the elbow plot on. If None, a new subplot is created.
        n_clusters : int, optional
            The number of clusters to use when computing the Cophenetic Correlation. The default value is 4.
        """
        coph_cor = pd.DataFrame({
            "Rank": self.get_ranks(),
            "Cophenetic Correlation": [self.cophenetic_correlation(self.consensus_matrix(rank, n_clusters=n_clusters))
                                       for rank in self.get_ranks()]
        })
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))
        sns.pointplot(coph_cor, x='Rank', y='Cophenetic Correlation', color='m', ax=ax)
        ax.set_ylim(min(coph_cor['Cophenetic Correlation']) - 0.05, 1)
        plt.show()

    def gene_cophenetic_correlation_elbow_plot(self, ax=None):
        """
        Elbow plot of the Cophenetic Correlation for gene factors.

        Parameters:
        -----------
        ax: matplotlib axes object, optional
            The axes to plot on. If not given, a new figure is created.
        """
        coph_cor = pd.DataFrame({
            "Rank": self.get_ranks(),
            "Cophenetic Correlation": [self.cophenetic_correlation(self.gene_consensus_matrix(rank=rank)) for rank in
                                       self.get_ranks()]
        })
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))
        sns.pointplot(coph_cor, x='Rank', y='Cophenetic Correlation', color='m', ax=ax)
        ax.set_ylim(min(coph_cor['Cophenetic Correlation']) - 0.05, 1)
        plt.show()

    def cophonetic_correlation_heatmap(self, rank: int, n_clusters: int = 4, size: int = 3):
        """
        Heatmap visualizing cophenetic correlation at a rank.

        Parameters:
        -----------
        rank: int
            The rank to be used.
        n_clusters: int, optional (default=4)
            The number of clusters to use when computing the consensus matrix.
        size: int, optional (default=3)
            The size of the heatmap.
        """
        C = self.consensus_matrix(rank, n_clusters=n_clusters)
        self.consensus_plot(1 - C, size=size)
        plt.show()

    def plot_gene_across_components(self, rank: int, gene: str, restart_index: int, figsize=(6, 4), color='k'):
        """
        Plot mean and high density region of the loadings of a given gene across the factors.

        Parameters:
        -----------
        rank: int
            The rank of the factorization.
        gene: str
            The name of the gene to be plotted.
        restart_index: int
            The index of the restart to be used.
        figsize: tuple of int, optional (default=(6, 4))
            The size of the figure.
        color: str, optional (default='k')
            The color of the scatter plot and vertical lines.

        Returns:
        --------
        fig: matplotlib figure object
            The resulting figure.
        """
        gene_index = self.sc_tensor.gene_list.index(gene)
        factorization = self.get_factorization(rank, restart_index)
        gene_factors = factorization.gene_factor['mean']
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.scatterplot(x=torch.arange(gene_factors.shape[1]), y=gene_factors[gene_index, :], s=50, color=color,
                             ax=ax)
        ax.set(xlabel="Component", ylabel="Loading", title=gene)
        ax.vlines(torch.arange(gene_factors.shape[1]),
                  factorization.gene_factor['|0.89'][gene_index, :],
                  factorization.gene_factor['0.89|'][gene_index, :], color=color, linewidth=3.0)
        plt.ylabel("Loading", labelpad=10)
        plt.tick_params(axis='x', which='major', pad=10)
        plt.tick_params(axis='y', which='major', pad=5)
        return fig

    def rank_metrics_plot(self, force=False, max_parallel_threads=1, entropy=-1000, eps=-1000):
        """
        Plots various rank metrics including Explained Variance, Gene Cophenetic Correlation, and Silhouette Score.

        Parameters
        ----------
        force : bool, optional
            If True, then the gene consensus matrix will be recomputed even if it already exists. Default is False.
        max_parallel_threads : int, optional
            The maximum number of parallel threads to use. Default is 1.
        entropy : float, optional
            The entropy value to use for computing the gene consensus matrix. Default is -1000.
        eps : float, optional
            The epsilon value to use for computing the gene consensus matrix. Default is -1000.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the rank metrics plot.
        """
        ranks = list(self.get_ranks())

        if 'gene_consensus_lax' not in dir(self) or self.gene_consensus_lax is None or force:
            consensus_partial = lambda rank: self.gene_consensus_matrix(rank, entropy=entropy, eps=eps)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_threads) as executor:
                self.gene_consensus_lax = dict(zip(ranks, executor.map(consensus_partial, ranks)))

        var_explained = pd.DataFrame({
            rank: [self.variance_explained(rank, i).item() for i in range(len(self.factorizations[rank].keys()))]
            for rank in ranks
        })
        coph_cor = pd.DataFrame({
            "Rank": ranks,
            "Cophenetic Correlation": [self.cophenetic_correlation(self.gene_consensus_lax[rank]) for rank in ranks]
        })

        all_cluster_metrics = self.cluster_gene_factors()

        silhouette = [all_cluster_metrics[ranks[i]].iloc[i]['silhouette_score'] for i in range(len(ranks))]

        fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharey=False)
        axs[0].errorbar(var_explained.mean(0).index, var_explained.mean(0).values, yerr=var_explained.std(0).values,
                        fmt='.-', color='r', mfc='k', mec='k')
        axs[0].set_ylim(min(var_explained.mean(0).values) - 0.05, 1.03)
        axs[0].set_title('Explained Variance', fontsize=10)
        axs[0].set_xlabel('Rank')

        axs[1].plot(coph_cor['Rank'], coph_cor['Cophenetic Correlation'], '.-', color='r', mfc='k', mec='k')
        axs[1].set_ylim(min(coph_cor['Cophenetic Correlation']) - 0.05, 1.03)
        axs[1].set_title('Gene Cophenetic Correlation', fontsize=10)
        axs[1].set_xlabel('Rank')

        axs[2].plot(ranks, silhouette, '.-', color='r', mfc='k', mec='k')
        axs[2].set_title('Silhouette Score', fontsize=10)
        axs[2].set_xlabel('Number of clusters')
        axs[2].set_ylim(min(silhouette) - 0.1, 1.05)

        fig.tight_layout()
        plt.show()
        return fig

    def gene_consensus_matrix_plots(self, force=False):
        """
        Plots the consensus matrix for each rank of the factorization.

        Parameters
        ----------
        force : bool, optional
            If True, forces the generation of the consensus matrix even if it has already been calculated.
            Default is False.
        """
        if 'all_gene_consensus_matrix' not in dir(self) or self.all_gene_consensus_matrix is None or force:
            self.all_gene_consensus_matrix = {}
            for rank in self.get_ranks():
                self.all_gene_consensus_matrix[rank] = self.gene_consensus_matrix(rank=rank)

        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for ax, rank in zip(axs.flatten(), self.factorizations.keys()):
            self.consensus_plot(1 - self.all_gene_consensus_matrix[rank], size=3, ax=ax)
            ax.set_title(f'Rank = {rank}', fontsize=10)
        fig.tight_layout()
        plt.show()

    def cluster_gene_factors(self, force=False):
        """
        Performs clustering on the gene factors and returns a dictionary of clustering metrics.

        Parameters
        ----------
        force : bool, optional
            If True, forces the recalculation of the clustering metrics even if they have already been calculated.
            Default is False.

        Returns
        -------
        all_cluster_metrics : dict
            A dictionary with ranks as keys and clustering metrics as values. The clustering metrics include
            'n_clusters', 'inertia', and 'silhouette_score'.
        """
        if 'all_cluster_metrics' in dir(self) and self.all_cluster_metrics is not None and not force:
            return self.all_cluster_metrics

        self.all_cluster_metrics = {}

        for rank in self.get_ranks():
            data = np.column_stack([f.gene_factor['mean'].numpy() for f in self.factorizations[rank].values()]).T
            sums = data.T.sum(axis=0)
            sums[sums == 0] = 1
            data_normed = (data.T / sums).T * 1e5

            cluster_metrics = pd.DataFrame(columns=['n_clusters', 'inertia', 'silhouette_score'])
            for n_clusters in self.get_ranks():
                if n_clusters < data_normed.shape[0]:
                    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, tol=1e-8)
                    estimator = make_pipeline(StandardScaler(), kmeans).fit(data_normed)
                    cluster_metrics.loc[len(cluster_metrics.index)] = [
                        n_clusters, estimator[-1].inertia_,
                        metrics.silhouette_score(data_normed, estimator[-1].labels_)
                    ]
            self.all_cluster_metrics[rank] = cluster_metrics
        return self.all_cluster_metrics

    def plot_gene_factor_cluster_metrics_per_rank(self, rank):
        """
        Plots the cluster metrics for the given rank.

        Parameters
        ----------
        rank : int
            Rank of the factorization.

        Returns
        -------
        fig : Figure
            The figure object.
        """
        cluster_metrics = self.cluster_gene_factors()[rank]

        fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
        axs[0].plot(cluster_metrics['n_clusters'], cluster_metrics['inertia'], 'r.-')
        axs[0].set_title('Inertia', fontsize=10)
        axs[0].set_xlabel('Number of clusters')

        axs[1].plot(cluster_metrics['n_clusters'], cluster_metrics['silhouette_score'], 'y.-')
        axs[1].set_title('Silhouette Score', fontsize=10)
        axs[1].set_xlabel('Number of clusters')

        fig.tight_layout()
        plt.show()
        return fig

    def reconstruct_factors_from_median_gene_factor(self, rank, n_clusters=None):
        """
        Reconstructs factors from the median gene factor for the given rank.

        Parameters
        ----------
        rank : int
            Rank of the factorization.
        n_clusters : int or None, optional
            Number of clusters to use for KMeans clustering, default is to use n_clusters = rank.

        Returns
        -------
        medians : ndarray
            An array containing the reconstructed factors.
        """
        if n_clusters is None:
            n_clusters = rank
        data = np.column_stack([f.gene_factor['mean'].numpy() for f in self.factorizations[rank].values()]).T
        sums = data.T.sum(axis=0)
        sums[sums == 0] = 1
        data_normed = (data.T / sums).T * 1e5
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=20, tol=1e-8)
        labels_ = make_pipeline(StandardScaler(), kmeans).fit(data_normed)[-1].labels_

        medians = np.stack(
            np.median(data_normed[np.where(labels_ == group)[0], :], axis=0) for group in np.unique(labels_))

        return medians

    def plot_components(self,
                        rank: int,
                        restart_index: int,
                        sample_grouping_column: str = 'sample_type',
                        plot_erichment_terms: bool = False,
                        enrich_dbs: list[str] = ['GO_Biological_Process_2021'],  # ,'KEGG_2021_Human', 'Reactome_2022'
                        normalize_gene_factors: bool = True,
                        threshold: float = 0.8,
                        sort_by: str = 'entropy',
                        entropy: int = 3,
                        eps: int = 0,
                        alpha=0.9,
                        top=0.9,
                        factors=None,
                        title=True
                        ):
        """Create a seaborn facetgrid summarizing the results of a CP decomposition.

        Parameters
        ----------
        rank : int
            The rank of the CP decomposition.
        restart_index : int
            The index of the CP decomposition restart to use.
        sample_grouping_column : str, optional
            The column name to group the samples by, by default 'sample_type'.
        plot_erichment_terms : bool, optional
            Whether to plot the top enriched terms for each gene factor, by default False.
        enrich_dbs : list of str, optional
            The gene sets to use for enrichment analysis, by default ['GO_Biological_Process_2021'].
        normalize_gene_factors : bool, optional
            Whether to normalize the gene factors, by default True.
        threshold : float, optional
            The threshold for selecting genes based on gene factor loadings, by default 0.8.
        sort_by : str, optional
            The metric to use for sorting genes based on gene factor loadings, by default 'entropy'.
        entropy : int, optional
            The number of top entropy terms to select, by default 3.
        eps : int, optional
            The number of genes to add to each gene factor, by default 0.
        alpha : float, optional
            The transparency level of the plotted points, by default 0.9.
        top : float, optional
            The space above the top row of the grid, by default 0.9.
        factors : list, optional
            The factors to include in the plot, by default None.
        title : bool, optional
            Whether to include a title for the plot, by default True.

        Returns
        -------
        sns.axisgrid.FacetGrid
            A seaborn facetgrid object.
        """
        labeled_sample_factor = self.get_labeled_sample_factor(rank, restart_index, sample_grouping_column, True)
        labeled_sample_factor['type'] = labeled_sample_factor[sample_grouping_column]
        labeled_sample_factor['mode'] = 'sample'

        labeled_celltype_factor = self.get_labeled_celltype_factor(rank, restart_index, True)
        labeled_celltype_factor['mode'] = 'celltype'

        if self.get_factorization(rank, restart_index).region_factor is not None:
            labeled_region_factor = self.get_labeled_region_factor(rank, restart_index, True)
            labeled_region_factor['mode'] = 'region'
            labeled_factors = pd.concat([labeled_sample_factor, labeled_celltype_factor, labeled_region_factor])
        else:
            labeled_factors = pd.concat([labeled_sample_factor, labeled_celltype_factor])
        labeled_factors = labeled_factors[['mode', 'type', 'factor', 'index', 'value']]

        gene_programs = self.get_gene_programs(rank, restart_index, normalize_gene_factors=normalize_gene_factors,
                                               threshold=threshold, sort_by=sort_by, entropy=entropy, eps=eps)

        for factor, program in gene_programs.items():
            for i in range(min(len(program), 20)):
                labeled_factors.loc[len(labeled_factors.index)] = ['gene', '.', str(factor), i, program[i]]
            if plot_erichment_terms:
                enr = gp.enrichr(gene_list=program, gene_sets=enrich_dbs, organism='human', background=self.sc_tensor.gene_list, outdir=None)
                top_terms = '\n '.join(enr.results['Term'][:3])
                labeled_factors.loc[len(labeled_factors.index)] = ['gene set enrichment', '.', str(factor), 0,
                                                                   top_terms]

        if factors is not None:
            labeled_factors = labeled_factors[labeled_factors['factor'].isin(factors)]

        g = sns.FacetGrid(labeled_factors, row='factor', col='mode', hue='type', height=1.5, aspect=2.5, sharex='col',
                          sharey=False, palette='tab20')
        g = g.map(self.mapplot, 'index', 'value', alpha=alpha, width=1)
        g.figure.subplots_adjust(wspace=0.2)
        g.set_xticklabels(rotation=90)
        plt.subplots_adjust(top=top)
        if title: g.fig.suptitle('CP Decomposition Factors')
        g.add_legend()
        plt.show()
        return g

    def get_labeled_sample_factor(self, rank: int, restart_index: int, grouping_column: str = 'sample_type',
                                  melt: bool = False):
        """
        Returns a labeled sample factor dataframe

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The index of the restart.
        grouping_column : str, optional
            The name of the column to group by, by default 'sample_type'.
        melt : bool, optional
            Whether to melt the dataframe, by default False.

        Returns
        -------
        pd.DataFrame
            A labeled sample factor dataframe.
        """
        factorization = self.get_factorization(rank, restart_index)
        dH = pd.DataFrame(factorization.sample_factor['mean'])
        dH = pd.concat([dH.reset_index(drop=True), self.sc_tensor.sample_features.reset_index(drop=True)], axis=1)
        dH.columns = dH.columns.map(str)
        # dH['index'] = list(range(dH.shape[0]))
        dH['index'] = self.sc_tensor.sample_list
        dH = dH.sort_values(by=grouping_column)
        if melt:
            dH = dH.melt(id_vars=self.sc_tensor.sample_features.columns.tolist() + ['index'], var_name='factor',
                         value_name='value')
        return dH

    def get_labeled_celltype_factor(self, rank: int, restart_index: int, melt: bool = False):
        """
        Returns a labeled celltype factor dataframe

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The index of the restart.
        melt : bool, optional
            Whether to melt the dataframe, by default False.

        Returns
        -------
        pd.DataFrame
            A labeled celltype factor dataframe.
        """
        dH = pd.DataFrame(self.get_factorization(rank, restart_index).celltype_factor['mean'])
        dH.columns = dH.columns.map(str)
        dH['type'] = self.sc_tensor.celltype_list
        dH['index'] = self.sc_tensor.celltype_list
        if melt:
            dH = dH.melt(id_vars=['type', 'index'], var_name='factor', value_name='value')
        return dH

    def get_labeled_region_factor(self, rank: int, restart_index: int, melt: bool = False):
        """
        Returns a labeled region factor dataframe

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The index of the restart.
        melt : bool, optional
            Whether to melt the dataframe, by default False.

        Returns
        -------
        pd.DataFrame
            A labeled region factor dataframe.
        """
        dH = pd.DataFrame(self.get_factorization(rank, restart_index).region_factor['mean'])
        dH['type'] = self.sc_tensor.region_list
        dH.columns = dH.columns.map(str)
        dH['index'] = self.sc_tensor.region_list
        if melt:
            dH = dH.melt(id_vars=['type', 'index'], var_name='factor', value_name='value')
        return dH

    def sample_factor_correlation(self, rank, restart_index, x, y, order, pairs):
        """
        Boxplot to show the correlation between sample feature and factor loadings

        Parameters
        ----------
        rank : int
            Rank of the tensor.
        restart_index : int
            Index of the restart.
        x : str
            The sample feature to be plotted on x-axis.
        y : int
            The factor loading to be plotted on y-axis.
        order : list
            The order in which the samples should be plotted.
        pairs : list
            The sample pairs to be annotated.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting figure object.
        """
        sample_factors = pd.concat([self.sc_tensor.sample_features,
                                    pd.DataFrame(self.get_factorization(rank, restart_index).sample_factor['mean'],
                                                 index=self.sc_tensor.sample_features.index)], axis=1)

        fig, ax = plt.subplots(figsize=(3, 4))
        ax = sns.boxplot(data=sample_factors, x=x, y=y, width=0.5, order=order, hue_order=order, showcaps=False,
                         flierprops={"marker": "."}, palette='tab20', hue=x, dodge=False)

        plt.legend(frameon=False)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_title(f'Factor {y}', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Loading')

        annotator = Annotator(ax, pairs, data=sample_factors, x=x, y=y, order=order)
        annotator.configure(test='Mann-Whitney', text_format='star')
        annotator.apply_and_annotate()
        return fig

    def get_sample_feature_correlated_factors(self, rank, restart_index, sample_feature, threshold=0.01,
                                              celltype_threshold=0.4):
        """
        Retrieves  factors that are significantly correlated with the given sample feature.

        Parameters
        ----------
        rank : int
            Rank of the tensor.
        restart_index : int
            Index of the restart.
        sample_feature : str
            The sample feature to correlate with the factors.
        threshold : float, optional
            The threshold p-value for significance of correlation. Default is 0.01.
        celltype_threshold : float, optional
            The threshold value for cell type filtering. Default is 0.4.

        Returns
        -------
        sig_factors : pandas.DataFrame
            The data frame containing the significant factors.
            Columns:
                - Factor: the significant factor index
                - Type: the type of the significant sample feature
                - P value: the p-value of the Mann-Whitney U test
                - Celltypes: the significant cell types
        """
        sample_factors = pd.concat([self.sc_tensor.sample_features,
                                    pd.DataFrame(self.get_factorization(rank, restart_index).sample_factor['mean'],
                                                 index=self.sc_tensor.sample_features.index)], axis=1)

        sig_factors = pd.DataFrame(columns=['Factor', 'Type', 'P value', 'Celltypes'])
        for factor in range(rank):
            sub_values = sample_factors.groupby(sample_feature)[factor].apply(list).to_dict()
            values1, values2 = sub_values.values()
            type1, type2 = sub_values.keys()
            mwu = mannwhitneyu(values1, values2, method='exact')

            ctype_factors = self.get_labeled_celltype_factor(rank, restart_index)
            celltypes = ctype_factors['type'][
                ctype_factors[str(factor)] / max(ctype_factors[str(factor)]) > celltype_threshold].values

            if mwu[1] <= threshold:
                stype = type1 if np.mean(values1) > np.mean(values2) else type2
                sig_factors.loc[len(sig_factors.index)] = [str(factor), stype, mwu[1], celltypes]
        return sig_factors

    def get_gene_programs(self,
                          rank: int,
                          restart_index: int,
                          normalize_gene_factors: bool = True,
                          threshold: float = 0.8,
                          sort_by='entropy',
                          entropy: int = 3,
                          eps: int = 0,
                          return_argmx: bool = False
                          ):
        """Compute gene programs for a given rank and restart index.

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart_index : int
            The restart index of the factorization.
        normalize_gene_factors : bool, optional
            Whether to normalize the gene factors, by default True.
        threshold : float, optional
            The threshold for selecting features, by default 0.8.
        sort_by : str, optional
            The sorting order for the gene programs, by default 'entropy'.
        entropy : int, optional
            The entropy threshold for selecting features. Genes with entropy less than this value times the MAD away from
            the median entropy are filtered out. By default 3.
        eps : int, optional
            The absolute score threshold. Features whose max loading is less than this values times the SD away from the
            median max loading are filtered out. By default 0.
        return_argmx : bool, optional
            Whether to return the argmax values, by default False.

        Returns
        -------
        dict
            A dictionary of gene programs.

        """
        gene_factor = self.get_factorization(rank, restart_index).gene_factor['mean'].numpy()
        if normalize_gene_factors:
            gene_factor = nn.functional.normalize(torch.from_numpy(gene_factor), p=1, dim=0).numpy()
        gss, scores = FactorizationSet.select_features(gene_factor, entropy=entropy, eps=eps, return_scores=True)
        gscores = dict(zip(self.sc_tensor.gene_list, scores))
        selected_features = pd.DataFrame(gene_factor[gss, :], index=np.asarray(self.sc_tensor.gene_list)[gss])

        mm = (selected_features.T - selected_features.min(axis=1)) / (
                selected_features.max(axis=1) - selected_features.min(axis=1))
        mm = mm.T
        y, x = np.where(mm > threshold)
        gene_programs = pd.DataFrame({'y': selected_features.index[y].to_list(), 'x': x}).groupby('x')['y'].apply(
            list).to_dict()

        res = [pd.DataFrame({'gene': gene_prog,
                             'relative': [-mm.loc[g, f] for g in gene_prog],
                             'entropy': [-np.round(gscores[g], 1) for g in gene_prog],
                             'absolute': [-selected_features.loc[g, f] for g in gene_prog]}) for f, gene_prog in
               gene_programs.items()]
        gene_programs = {i: list(res[i].sort_values(by=['relative', 'entropy', 'absolute']).gene.values) for i in
                         range(len(gene_programs))}
        if sort_by == 'entropy':
            [gene_prog.sort(key=lambda g: -gscores[g]) for gene_prog in gene_programs.values()]
        # print([len(gene_prog) for gene_prog in gene_programs.values()])
        return (gene_programs, selected_features.idxmax(axis=1)) if return_argmx else gene_programs

    def gene_consensus_matrix(self, rank: int, normalize_gene_factors: bool = True,
                              threshold: float = 0.8, sort_by='entropy', entropy: int = 3, eps: int = 0,
                              max_parallel_threads: int = 32):
        """
        Compute consensus matrix of gene programs across restarts. The consensus matrix gives an indication of how often
        a gene is included in the gene program across all restarts, and can be used to identify genes that are
        consistently important in the factorization.

        Parameters
        ----------
        rank : int
            The rank of the factorization for which to compute the consensus matrix.
        normalize_gene_factors : bool, optional
            Whether to normalize gene factors (default is True).
        threshold : float, optional
            Threshold to use for determining gene program membership (default is 0.8).
        sort_by : {'entropy', 'magnitude'}, optional
            Sort genes by entropy or magnitude (default is 'entropy').
        entropy : int, optional
            The entropy threshold for selecting features. Genes with entropy less than this value times the MAD away from
            the median entropy are filtered out. By default 3.
        eps : int, optional
            The absolute score threshold. Features whose max loading is less than this values times the SD away from the
            median max loading are filtered out. By default 0.
        max_parallel_threads : int, optional
            Maximum number of threads to use for computing gene programs in parallel (default is 32).

        Returns
        -------
        numpy.ndarray
            Consensus matrix of gene programs across restarts.
        """

        restarts = self.factorizations[rank].keys()

        get_gene_programs_partial = lambda restart: self.get_gene_programs(
            rank, restart, normalize_gene_factors=normalize_gene_factors, threshold=threshold,
            sort_by=sort_by, entropy=entropy, eps=eps, return_argmx=True
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_threads) as executor:
            gps = dict(zip(restarts, executor.map(get_gene_programs_partial, restarts)))

        high_entropy_gene_set = set()
        all_argmx = [gps[restart][1] for restart in restarts]
        [high_entropy_gene_set.update(program) for restart in restarts for program in gps[restart][0].values()]

        HEG = pd.DataFrame(index=list(high_entropy_gene_set))
        for restart in restarts:
            argmx = all_argmx[restart].filter(items=high_entropy_gene_set, axis=0)
            HEG.loc[argmx.index, str(restart)] = argmx.values

        dim1 = HEG.shape[0]
        cons = np.mat(np.zeros((dim1, dim1)))
        for restart in restarts:
            mat1 = np.tile(HEG[str(restart)].values, (HEG.shape[0], 1))
            mat2 = np.tile(HEG[str(restart)].values.reshape((-1, 1)), (1, HEG.shape[0]))
            conn = np.mat(mat1 == mat2, dtype='d')
            cons += conn
        return np.divide(cons, len(restarts))

    def gene_factor_elbow_plots(self, rank, restart_index, factor_index=None, ncols=5, num_genes=20, fontsize=9,
                                normalize=True, figsize=(3.5, 3.5)):
        """
        Creates elbow plots for gene factors in a given rank and restart index.

        Parameters
        ----------
        rank : int
            Rank of the factorization.
        restart_index : int
            Index of the specific run.
        factor_index : int, optional
            Index of the factor to plot, default is None.
        ncols : int, optional
            Number of columns in the grid of plots, default is 5.
        num_genes : int, optional
            Number of genes to plot, default is 20.
        fontsize : int, optional
            Font size for gene labels, default is 9.
        normalize : bool, optional
            Whether to normalize the gene factor matrix, default is True.
        figsize : tuple of int, optional
            Size of the figure, default is (3.5, 3.5).

        Returns
        -------
        fig : matplotlib Figure object
            The created figure object.
        factors : pandas DataFrame
            The gene factor matrix.

        """
        gene_factor = self.get_factorization(rank=rank, restart_index=restart_index).gene_factor['mean'].numpy()
        if normalize:
            gene_factor = nn.functional.normalize(torch.from_numpy(gene_factor), p=2, dim=0).numpy()
            gene_factor = (gene_factor.T / gene_factor.T.sum(axis=0)).T
        factors = pd.DataFrame(gene_factor, index=self.sc_tensor.gene_list).fillna(0)

        if factor_index is not None:
            fig, ax = plt.subplots(figsize=figsize)
            load = factors[factor_index].sort_values(ascending=False)[:num_genes]
            ax.plot(load.values, '.', markersize=0)
            [ax.text(i, load[i], load.index[i], rotation='vertical', verticalalignment='bottom',
                     horizontalalignment='center', fontsize=fontsize) for i in range(num_genes)]
            ax.set_ylim(min(load.values) - max(load.values)*0.05, max(load.values)*1.15)
            ax.set_xlim(-2, num_genes)
            ax.set_xticks([])
            ax.set_title(f'Factor {factor_index}')
            ax.set_xlabel('ranking')
            ax.set_ylabel('score')

            fig.tight_layout()
            plt.show()
            return fig, factors

        nrows = (rank // ncols) + 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))

        for ax, factor_index in zip(axs.flatten(), range(rank)):
            load = factors[factor_index].sort_values(ascending=False)

            ax.plot(load[:num_genes].values, '.', markersize=0)
            [ax.text(i, load[i], load.index[i], rotation='vertical', verticalalignment='bottom',
                     horizontalalignment='center', fontsize=fontsize) for i in range(num_genes)]
            ax.set_ylim(min(load[:num_genes].values) - max(load[:num_genes].values)*0.05, max(load[:num_genes].values)*1.15)
            ax.set_xlim(-2, num_genes)
            ax.set_xticks([])
            ax.set_title(f'Factor {factor_index}')
            if rank - factor_index <= ncols:
                ax.set_xlabel('ranking')

            if factor_index % ncols == 0:
                ax.set_ylabel('score')
        fig.tight_layout()
        [fig.delaxes(ax) for ax in axs.flatten()[rank:]]
        plt.show()
        return fig, factors

    def save_rank_results(self, rank, restart=0, dir_path=None, entropy=1, eps=0,
                          sample_grouping_column='sample_type', plot_erichment_terms=True):
        """
        Save rank results.

        Parameters
        ----------
        rank : int
            The rank of the factorization.
        restart : int, optional
            The index of the factorization run to use.
        dir_path : str, optional
            The directory path to save the results to.
        entropy : int, optional
            The entropy threshold for selecting features. Genes with entropy less than this value times the MAD away from
            the median entropy are filtered out. By default 3.
        eps : int, optional
            The absolute score threshold. Features whose max loading is less than this values times the SD away from the
            median max loading are filtered out. By default 0.
        sample_grouping_column : str, optional
            The column in the adata object to use for sample grouping.

        """
        if dir_path is None: dir_path = f"results/rank_{rank}"
        os.makedirs(dir_path, exist_ok=True)

        gene_programs_stringent = self.get_gene_programs(
            rank, restart, normalize_gene_factors=True, threshold=0.99, sort_by='other', entropy=entropy, eps=eps
        )
        mx = max(len(v) for v in gene_programs_stringent.values())
        df = pd.DataFrame({k: v + [np.nan] * (mx - len(v)) for k, v in gene_programs_stringent.items()})
        df.to_csv(os.path.join(dir_path, f"rank_{rank}_gene_programs_stringent.csv"))

        gene_programs = self.get_gene_programs(
            rank, restart, normalize_gene_factors=True, threshold=0.6, sort_by='other', entropy=entropy, eps=eps
        )
        mx = max(len(v) for v in gene_programs.values())
        df = pd.DataFrame({k: v + [np.nan] * (mx - len(v)) for k, v in gene_programs.items()})
        df.to_csv(os.path.join(dir_path, f"rank_{rank}_gene_programs.csv"))

        gene_factor = self.get_factorization(rank=rank, restart_index=restart).gene_factor['mean'].numpy()
        df = pd.DataFrame(gene_factor, index=self.sc_tensor.gene_list)
        df.to_csv(os.path.join(dir_path, f"rank_{rank}_raw_gene_factor_loadings.csv"))

        fig = self.plot_components(
            rank=rank, restart_index=restart, threshold=0.6, entropy=entropy, eps=eps, sort_by='other',
            sample_grouping_column=sample_grouping_column, title=False,
            plot_erichment_terms=plot_erichment_terms, normalize_gene_factors=True
        )
        fig.savefig(os.path.join(dir_path, f"rank_{rank}_factors.png"), dpi=200, bbox_inches="tight")

        rich.print(f"Saved gene program for rank {rank} at {dir_path}")

    def save(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Saved SingleCellTensor object to path {path}")

    def __repr__(
            self,
    ):
        rich.print(f"Single cell factorization object with the following params:"
                   f"\n\tTensor size {' x '.join([str(i) for i in self.sc_tensor.tensor.shape])}"
                   f"\n\tRanks: {list(self.get_ranks())}")
        return ""

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def select_features(W, entropy=3, eps=0, return_scores=False):
        """
        Select high entropy features that differentiate basis vectors in `W`.

        Parameters
        ----------
        entropy : int, optional
            The entropy threshold for selecting features. Genes with entropy less than this value times the MAD away from
            the median entropy are filtered out. By default 3.
        eps : int, optional
            The absolute score threshold. Features whose max loading is less than this values times the SD away from the
            median max loading are filtered out. By default 0.
        return_scores : bool, optional
            Flag indicating whether to return the scores or not, by default False.

        Returns
        -------
        numpy.ndarray or tuple of (numpy.ndarray, numpy.ndarray)
            Boolean array of shape (n_features,) indicating the selected features,
            or a tuple containing the selected features and their scores.
        """

        # scores = np.zeros(W.shape[0])
        # for f in range(W.shape[0]):
        #     # probability that the i-th feature contributes to q-th basis vector.
        #     prob = W[f, :] / (W[f, :].sum() + np.finfo(W.dtype).eps)
        #     scores[f] = np.dot(prob, np.log2(prob + np.finfo(W.dtype).eps).T)
        # scores = 1. + 1. / np.log2(W.shape[1]) * scores

        prob = W / (np.sum(W, axis=1, keepdims=True) + np.finfo(W.dtype).eps)
        scores = np.sum(prob * np.log2(prob + np.finfo(W.dtype).eps), axis=1)
        scores = 1. + 1. / np.log2(W.shape[1]) * scores

        entropy_threshold = np.median(scores) + entropy * np.median(abs(scores - np.median(scores)))
        max_loading_threshold = np.median(W.flatten()) + (eps * np.std(W.flatten()))
        selected = (scores > entropy_threshold) & (np.max(W, axis=1) > max_loading_threshold)
        return (selected, scores) if return_scores else selected

    @staticmethod
    def connectivity(sample_factor, n_clusters=4):
        """
        Compute the connectivity matrix based on the clusters assigned to the samples in `sample_factor`.

        Parameters
        ----------
        sample_factor : numpy.ndarray
            Input matrix of shape (n_samples, n_features).
        n_clusters : int, optional
            Number of clusters, by default 4.

        Returns
        -------
        numpy.matrix
            Binary connectivity matrix of shape (n_samples, n_samples).
        """
        idx = KMeans(init="k-means++", n_clusters=n_clusters, random_state=20221021).fit(sample_factor).labels_
        mat1 = np.tile(idx, (sample_factor.shape[0], 1))
        mat2 = np.tile(idx.reshape((-1, 1)), (1, sample_factor.shape[0]))
        conn = mat1 == mat2
        return np.mat(conn, dtype='d')

    @staticmethod
    def cophenetic_correlation(consensus):
        """
        Compute the cophenetic correlation coefficient of a consensus matrix.

        The cophenetic distance between two leaves of a tree is the height of the closest node that leads to
        both leaves. If a clustering is good, the cophenetic distance will be well correlated with the
        original distance from the dissimilarity matrix.

        Parameters
        ----------
        consensus : numpy.ndarray
            Consensus matrix of shape (n_samples, n_samples).

        Returns
        -------
        float
            The cophenetic correlation coefficient.
        """
        I_minus_C = np.asarray(1 - consensus)
        upper_triangular_I_minus_C = I_minus_C[np.triu_indices(I_minus_C.shape[0], k=1)]
        Y = fastcluster.linkage(upper_triangular_I_minus_C, method='average')
        cophenetic_dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(Y))
        upper_triangular_cophenetic_dist = cophenetic_dist[np.triu_indices(cophenetic_dist.shape[0], k=1)]
        return scipy.stats.pearsonr(upper_triangular_I_minus_C, upper_triangular_cophenetic_dist)[0]

    @staticmethod
    def mapplot(x, y, **kwargs):
        try:
            if type(y.values[0]) == str:
                strt = ', '.join([x + '\n' if (i + 1) % 5 == 0 else x for i, x in enumerate(y)]).replace('\n,', ',\n')
                plt.text(-0.1, 0.9, strt, ha='left', va='top', size=10)
                plt.axis('off')
            else:
                plt.bar(x, y, **kwargs)
        except:
            pass

    @staticmethod
    def consensus_plot(C, plot_dendrogram=False, summary_df=None, size=5, ax=None):
        """
        Plot a consensus matrix.

        Parameters
        ----------
        C : array_like
            A square matrix representing the consensus matrix.
        plot_dendrogram : bool, optional
            If True, a dendrogram will be plotted. Default is False.
        summary_df : pandas.DataFrame, optional
            A pandas dataframe containing the summary of data. Default is None.
        size : int, optional
            The size of the plot. Default is 5.
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. Default is None.

        Returns
        -------
        Y : ndarray
            An ndarray representing the linkage matrix.
        """

        def clean_axis(ax, spines=False):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            for sp in ax.spines.values():
                sp.set_visible(spines)

        label_names = summary_df.columns if summary_df is not None else []
        label_collection = [summary_df[cat].values for cat in label_names]
        wr = [0.25 * plot_dendrogram, 1] + [0.05] * len(label_collection)

        fig = plt.figure(figsize=(sum(wr) * size + len(wr) * 0.1, size)) if ax == None else ax
        heatmapGS = gridspec.GridSpec(1, 2 + len(label_collection), wspace=.01, hspace=0.01, width_ratios=wr)
        C = np.asarray(C)
        Y = scipy.cluster.hierarchy.linkage(C[np.triu_indices(C.shape[0], k=1)], method='average')

        if plot_dendrogram:
            denAX = fig.add_subplot(heatmapGS[0, 0])
        denD = scipy.cluster.hierarchy.dendrogram(Y, orientation='left')
        if plot_dendrogram:
            clean_axis(denAX)

        heatmapAX = fig.add_subplot(heatmapGS[0, 1]) if ax == None else ax
        D = C[denD['leaves'], :][:, denD['leaves']]
        axi = heatmapAX.imshow(-D, interpolation='nearest', aspect='equal', origin='lower', cmap="Greens")
        clean_axis(heatmapAX, spines=True)

        if len(label_collection) > 0:
            for c, labels in enumerate(label_collection):
                catAX = fig.add_subplot(heatmapGS[0, 2 + c])
                catAX.margins(x=0.05)
                categories = set(labels)
                colmap = sns.color_palette()
                color_for_cluster = {cat: colmap[i] for i, cat in enumerate(categories)}
                for i, cluster in enumerate(labels[denD['leaves']]):
                    catAX.add_patch(plt.Rectangle((c / len(label_names), i / len(labels)), 1, 1 / len(labels),
                                                  facecolor=color_for_cluster[cluster], edgecolor='k',
                                                  linewidth=0.2, clip_on=False))

                catAX.text((c + 0.5) / len(label_names), 0, label_names[c], fontsize=10, rotation='vertical', va='top')
                clean_axis(catAX)

        return Y

    @staticmethod
    def case_control(adata, gene, celltype, iv, case, donor_label, celltype_label, ax,
                     title=None, xlab=None, annot=False, stat_test=False):
        obs_pb = adata.obs.copy()

        obs_pb[gene] = adata.raw.X[:, adata.var_names.to_list().index(gene)].toarray().squeeze()
        df = obs_pb.groupby(by=[donor_label, celltype_label, iv]).agg({gene: 'mean'}).reset_index()
        df['case_control'] = 'Other'
        df.loc[df.loc[df[celltype_label] == celltype].loc[df[iv] == case].index, 'case_control'] = celltype + "_" + case
        x = 'case_control'
        y = gene

        ax = sns.swarmplot(data=df, x=x, y=y, palette='Set2', size=5, color="1", edgecolor='black', linewidth=1, ax=ax,
                           order=[celltype + "_" + case, 'Other'])

        ax = sns.boxplot(data=df, x=x, y=y, width=0.5, showcaps=False, boxprops={"edgecolor": 'black', "linewidth": 1},
                         flierprops={"marker": "."}, palette='Set2', hue=x, dodge=False, ax=ax,
                         order=[celltype + "_" + case, 'Other'], hue_order=[celltype + "_" + case, 'Other'])

        ax.get_legend().remove()
        ax.set_title(f'       {y}   in   {celltype}' if title is None else title)
        ax.set_xlabel('')
        ax.set_ylabel('Mean Raw Expression' if xlab is None else xlab)
        ax.tick_params(axis='x', labelrotation=90)

        if annot:
            annotator = Annotator(ax, data=df, pairs=[("HL", "RLN")], x=x, y=y)
            annotator.configure(test='Mann-Whitney', text_format='star')
            annotator.apply_and_annotate()
        if stat_test:
            annotator = Annotator(ax, data=df, pairs=[(celltype + "_" + case, "other")], x=x, y=y)
            annotator.configure(test='Mann-Whitney', text_format='star')
            annotator.apply_and_annotate()

    @staticmethod
    def case_control_1v1(adata, gene, celltype, iv, donor_label, celltype_label, ax,
                         title=None, xlab=None, annot=False):
        obs_pb = adata.obs.copy()
        obs_pb[gene] = adata.raw.X[:, adata.var_names.to_list().index(gene)].toarray().squeeze()
        df = obs_pb[obs_pb[celltype_label] == celltype].sort_values(by=iv)
        df = df.groupby(by=[donor_label, celltype_label]).agg({gene: 'mean', iv: 'first'}).reset_index()
        x, y = iv, gene
        ax = sns.swarmplot(data=df, x=x, y=y, palette='Set2', size=5, color="1", edgecolor='black', linewidth=1, ax=ax)
        ax = sns.boxplot(data=df, x=x, y=y, width=0.5, showcaps=False, boxprops={"edgecolor": 'black', "linewidth": 1},
                         flierprops={"marker": "."}, palette='Set2', hue=x, dodge=False, ax=ax)
        ax.get_legend().remove()
        ax.set_title(f'       {y}   in   {celltype}' if title is None else title)
        ax.set_xlabel('')
        ax.set_ylabel('Mean Raw Expression' if xlab is None else xlab)
        if annot:
            annotator = Annotator(ax, data=df, pairs=[("HL", "RLN")], x=x, y=y)
            annotator.configure(test='Mann-Whitney', text_format='star')
            annotator.apply_and_annotate()

    @staticmethod
    def gene_box_plot(adata, gene, celltype, iv='condition', donor_label='donor', celltype_label='cell_types_level_3',
                      ptype='point'):
        obs_pb = adata.obs.copy()
        obs_pb[gene] = adata.X[:, adata.var_names.to_list().index(gene)].toarray().squeeze()
        df = obs_pb[obs_pb[celltype_label] == celltype].sort_values(by=iv)
        df[donor_label] = df[donor_label].astype(str)

        x, y = iv, gene
        fig, axs = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

        if ptype == 'box':
            sns.boxplot(data=df[df[iv] == np.unique(df[iv])[0]], x=donor_label, y=y, notch=True, width=0.5,
                        showcaps=False,
                        boxprops={"facecolor": sns.color_palette("Set2")[0]}, flierprops={"marker": "."}, dodge=False,
                        ax=axs[0])
            sns.boxplot(data=df[df[iv] == np.unique(df[iv])[1]], x=donor_label, y=y, notch=True, width=0.5,
                        showcaps=False,
                        boxprops={"facecolor": sns.color_palette("Set2")[1]}, flierprops={"marker": "."}, dodge=False,
                        ax=axs[1])
        elif ptype == 'point':
            sns.pointplot(data=df[df[iv] == np.unique(df[iv])[0]], x=donor_label, y=y, ax=axs[0],
                          color=sns.color_palette("Set2")[0], join=False,
                          estimator=np.mean, errorbar='sd', markers='.', scale=1.2, errwidth=2)
            sns.pointplot(data=df[df[iv] == np.unique(df[iv])[1]], x=donor_label, y=y, ax=axs[1],
                          color=sns.color_palette("Set2")[1], join=False,
                          estimator=np.mean, errorbar='sd', markers='.', scale=1.2, errwidth=2)

        plt.subplots_adjust(wspace=0.07)

        axs[0].set_xlim(-1.2, 10.2)
        axs[1].set_xlim(-1.2, 5.2)

        axs[0].tick_params(axis='x', labelrotation=90)
        axs[1].tick_params(axis='x', labelrotation=90)

        axs[0].set_title(' Hodgkin Lymphoma donors ')
        axs[1].set_title(' Healthy donors ')

        axs[1].get_yaxis().set_visible(False)

        plt.show()
