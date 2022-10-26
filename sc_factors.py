import pickle
from collections import defaultdict

import scipy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.cluster import KMeans

from sc_tensor import SingleCellTensor
from bayesian_parafac import BayesianCP


class Factorization:
    """ Single group of factors """

    def __init__(self, sample_factor, celltype_factor, gene_factor, region_factor=None):
        self.sample_factor = sample_factor
        self.celltype_factor = celltype_factor
        self.gene_factor = gene_factor
        self.region_factor = region_factor

        self.has_region = region_factor != None

    def get_means(self):
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
    """ A collection of factorizations indexed by rank and restart number """

    def __init__(self, sc_tensor: SingleCellTensor = None):

        self.sc_tensor = sc_tensor
        self.factorizations = defaultdict(dict)
        self.factorization_parameters = defaultdict(dict)

    def add_factorization(self, rank: int, restart_index: int, factorization: Factorization):
        self.factorizations[rank][restart_index] = factorization

    def get_factorization(self, rank: int, restart_index: int):
        return self.factorizations[rank][restart_index]

    def get_ranks(self):
        return self.factorizations.keys()

    def add_precis_factorization(self, rank: int, restart_index: int, precis: dict):
        if len(precis.keys()) == 3:
            self.add_factorization(
                rank, restart_index, Factorization(precis['factor_0'], precis['factor_1'], precis['factor_2']))
        else:
            raise Exception("Implement 4d factorization")

    def add_factorization_params(self, rank: int, restart_index: int, params: dict):
        self.factorization_parameters[rank][restart_index] = params

    def variance_explained(self, rank: int, restart_index: int):
        """ 1 - (t - t')^2 / || t ||^2 """
        factorization = self.get_factorization(rank, restart_index)
        tensor_means = torch.einsum(BayesianCP.get_einsum_formula(len(self.sc_tensor.tensor.shape)),
                                    *factorization.get_means())
        tensor = torch.from_numpy(self.sc_tensor.tensor)
        return 1 - (torch.norm(tensor.float() - tensor_means.float()) ** 2 / torch.norm(tensor.float()) ** 2)

    def variance_explained_elbow_plot(self, ax=None):
        """ Elbow plot of variance explained """
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
        """ Indicates stability across multiple factorization iterations """
        dim1 = self.get_factorization(rank, 0).sample_factor['mean'].shape[0]
        cons = np.mat(np.zeros((dim1, dim1)))
        for i in self.factorizations[rank].keys():
            sample_factor = self.get_factorization(rank, i).sample_factor['mean'].numpy()
            sample_factor = sample_factor / sample_factor.sum(axis=0)
            cons += FactorizationSet.connectivity(sample_factor=sample_factor, n_clusters=n_clusters)
        return np.divide(cons, len(self.factorizations[rank].keys()))

    def cophenetic_correlation(self, rank: int):
        """
        The cophenetic distance between two leaves of a tree is the height of the closest node that leads to
        both leaves. If a clustering is good, the cophenetic distance will be well correlated with the 
        original distance from the disimilarity matrix.
        """
        I_minus_C = np.asarray(1 - self.consensus_matrix(rank))
        upper_triangular_I_minus_C = I_minus_C[np.triu_indices(I_minus_C.shape[0], k=1)]
        Y = scipy.cluster.hierarchy.linkage(upper_triangular_I_minus_C, method='average')
        cophenetic_dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(Y))
        upper_triangular_cophenetic_dist = cophenetic_dist[np.triu_indices(cophenetic_dist.shape[0], k=1)]
        return scipy.stats.pearsonr(upper_triangular_I_minus_C, upper_triangular_cophenetic_dist)[0]

    def cophenetic_correlation_elbow_plot(self, ax=None, n_clusters: int = 4):
        """ Elbow plot of the Cophenetic Correlation """
        coph_cor = pd.DataFrame({
            "Rank": self.get_ranks(),
            "Cophenetic Correlation": [
                self.cophenetic_correlation(self.consensus_matrix(rank, n_clusters=n_clusters))
                for rank in self.get_ranks()
            ]
        })
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))
        sns.pointplot(coph_cor, x='Rank', y='Cophenetic Correlation', color='m', ax=ax)
        ax.set_ylim(min(coph_cor['Cophenetic Correlation']) - 0.05, 1)
        plt.show()

    def cophonetic_correlation_heatmap(self, rank: int, n_clusters: int = 4, size: int = 3):
        """ Heatmap visualizing cophenetic correlation at a rank """
        C = self.consensus_matrix(rank, n_clusters=n_clusters)
        self.consensus_plot(1 - C, size=size)
        plt.show()

    def plot_gene_across_components(self, rank: int, gene: str, restart_index: int):
        """ Plot mean and high density region of the loadings of a given gene across the factors """
        gene_index = self.sc_tensor.gene_list.index(gene)
        factorization = self.get_factorization(rank, restart_index)
        gene_factors = factorization.gene_factor['mean']
        with plt.rc_context({'figure.figsize': (4, 3)}):
            ax = sns.scatterplot(x=torch.arange(gene_factors.shape[1]), y=gene_factors[gene_index, :], s=40, color="k")
            ax.set(xlabel="Component", ylabel="Loading", title=gene)
            ax.vlines(torch.arange(gene_factors.shape[1]),
                      factorization.gene_factor['|0.89'][gene_index, :],
                      factorization.gene_factor['0.89|'][gene_index, :], color="k")
            plt.show()

    def plot_components(self,
                        rank: int,
                        restart_index: int,
                        sample_grouping_column: str = 'sample_type',
                        plot_erichment_terms: bool = False,
                        enrich_dbs: list[str] = ['GO_Biological_Process_2021'],  # ,'KEGG_2021_Human', 'Reactome_2022'
                        normalize_gene_factors: bool = True,
                        entropy: int = 3,
                        eps: int = 0
                        ):
        """ Create a seaborn facetgrid summarizing the results of a CP decomposition """

        factorization = self.get_factorization(rank, restart_index)

        dH = pd.DataFrame(factorization.gene_factor['mean'])
        dH = pd.concat([dH.reset_index(drop=True), self.sc_tensor.sample_features.reset_index(drop=True)], axis=1)
        dH.columns = dH.columns.map(str)
        dH = dH.sort_values(by=sample_grouping_column)
        dH['index'] = list(range(dH.shape[0]))
        dH = dH.melt(id_vars=self.sc_tensor.sample_features.columns.tolist() + ['index'], var_name='factor',
                     value_name='value')
        dH['mode'] = 'sample'
        dH['type'] = dH[sample_grouping_column]

        # ---

        dH2 = pd.DataFrame(factorization.celltype_factor['mean'])
        dH2.columns = dH2.columns.map(str)
        dH2['type'] = self.sc_tensor.celltype_list
        dH2['index'] = self.sc_tensor.celltype_list
        dH2 = dH2.melt(id_vars=['type', 'index'], var_name='factor', value_name='value')
        dH2['mode'] = 'celltype'

        # ---

        gene_factor = factorization.gene_factor['mean'].numpy()
        if normalize_gene_factors:
            gene_factor = nn.functional.normalize(torch.from_numpy(gene_factor)).numpy()
        gss, scores = FactorizationSet.select_features(gene_factor, entropy=entropy, eps=eps, return_scores=True)
        gscores = dict(zip(self.sc_tensor.gene_list, scores))
        selected_features = pd.DataFrame(gene_factor[gss, :], index=np.asarray(self.sc_tensor.gene_list)[gss])

        argmx = pd.DataFrame(selected_features.idxmax(axis=1))
        argmx = argmx.reset_index()
        argmx.columns = ['y', 'x']
        gene_programs = argmx.groupby('x')['y'].apply(list).to_dict()
        [gene_prog.sort(key=lambda g: -gscores[g]) for gene_prog in gene_programs.values()]
        print([len(gene_prog) for gene_prog in gene_programs.values()])

        # ---
        if factorization.region_factor is not None:
            dH3 = pd.DataFrame(factorization.region_factor['mean'])
            dH3['type'] = self.sc_tensor.region_list
            dH3.columns = dH3.columns.map(str)
            dH3['index'] = self.sc_tensor.region_list
            dH3 = dH3.melt(id_vars=['type', 'index'], var_name='factor', value_name='value')
            dH3['mode'] = 'region'

            df = pd.concat([dH, dH2, dH3])[['mode', 'type', 'factor', 'index', 'value']]
        else:
            df = pd.concat([dH, dH2])[['mode', 'type', 'factor', 'index', 'value']]

        # ---

        for factor, program in gene_programs.items():
            for i in range(min(len(program), 20)):
                df.loc[len(df.index)] = ['gene', '.', str(factor), i, program[i]]
            if plot_erichment_terms:
                enr = gp.enrichr(gene_list=program,
                                 gene_sets=enrich_dbs,
                                 organism='human', outdir=None)
                top_terms = '\n '.join(enr.results['Term'][:3])
                df.loc[len(df.index)] = ['gene set enrichment', '.', str(factor), 0, top_terms]

        g = sns.FacetGrid(df, row='factor', col='mode', hue='type', height=1.5, aspect=3, sharex='col', sharey=False)
        g = g.map(FactorizationSet.mapplot, 'index', 'value', alpha=0.6, width=1)
        g.figure.subplots_adjust(wspace=0.2)
        g.set_xticklabels(rotation=90)
        plt.subplots_adjust(top=0.90)
        g.fig.suptitle('CP Decomposition Factors')
        g.add_legend()
        plt.show()

        return gene_programs

    def save(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Saved SingleCellTensor object to path {path}")

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def select_features(W, entropy=3, eps=0, return_scores=False):
        """ Select high entropy features that differentiate basis vectors in `W` """
        scores = np.zeros(W.shape[0])
        for f in range(W.shape[0]):
            # probability that the i-th feature contributes to q-th basis vector.
            prob = W[f, :] / (W[f, :].sum() + np.finfo(W.dtype).eps)
            scores[f] = np.dot(prob, np.log2(prob + np.finfo(W.dtype).eps).T)
        scores = 1. + 1. / np.log2(W.shape[1]) * scores

        th = np.median(scores) + entropy * np.median(abs(scores - np.median(scores)))
        sel = scores > th
        m = np.median(W.tolist())
        sel = np.array([sel[i] and np.max(W[i, :]) > m + eps
                        for i in range(W.shape[0])])
        if return_scores:
            return sel, scores
        return sel

    @staticmethod
    def connectivity(sample_factor, n_clusters=4):
        idx = KMeans(init="k-means++", n_clusters=n_clusters, random_state=20221021).fit(sample_factor).labels_
        mat1 = np.tile(idx, (sample_factor.shape[0], 1))
        mat2 = np.tile(idx.reshape((-1, 1)), (1, sample_factor.shape[0]))
        conn = mat1 == mat2
        return np.mat(conn, dtype='d')

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
    def consensus_plot(C, plot_dendrogram=False, summary_df=None, size=5):
        """ Consensus matrix plot """

        def clean_axis(ax, spines=False):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            for sp in ax.spines.values():
                sp.set_visible(spines)

        label_names = summary_df.columns if summary_df is not None else []
        label_collection = [summary_df[cat].values for cat in label_names]
        wr = [0.25 * plot_dendrogram, 1] + [0.05] * len(label_collection)
        fig = plt.figure(figsize=(sum(wr) * size + len(wr) * 0.1, size))
        heatmapGS = gridspec.GridSpec(1, 2 + len(label_collection), wspace=.01, hspace=0.01, width_ratios=wr)
        C = np.asarray(C)
        Y = scipy.cluster.hierarchy.linkage(C[np.triu_indices(C.shape[0], k=1)], method='average')

        denAX = fig.add_subplot(heatmapGS[0, 0])
        denD = scipy.cluster.hierarchy.dendrogram(Y, orientation='left')
        clean_axis(denAX)

        heatmapAX = fig.add_subplot(heatmapGS[0, 1])
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


