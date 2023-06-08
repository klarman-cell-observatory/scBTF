import torch
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Union

from tensorly.decomposition import non_negative_parafac_hals

from scBTF.sc_tensor import SingleCellTensor
from scBTF.sc_factors import FactorizationSet
from scBTF.bayesian_parafac import BayesianCP

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


class SingleCellBTF:
    """ Bayesian Tensor Factorization of single cell rnaseq data """

    @staticmethod
    def factorize(
            sc_tensor: SingleCellTensor,
            rank: Union[int, list[int]],
            model: str = 'zero_inflated_poisson',
            n_restarts: int = 10,
            num_steps: int = 500,
            initial_lr: float = 1,
            init_alpha: float = 1e2,
            init_beta: float = 1.,
            fixed_mode_variance: float = 10.,
            fixed_mode: int = None,
            fixed_value=None,
            lr_decay_gamma: float = 1e-1,
            plot_var_explained: bool = True
    ) -> FactorizationSet:
        """ Run BTF on the tensor """

        if len(sc_tensor.tensor.shape) > 4:
            print('[Warning] tensor should be 3 or 4 dimensional!')

        tensor = torch.from_numpy(sc_tensor.tensor)
        # weights_init, factors_init = initialize_cp(tensor, non_negative=True, init='random', rank=10)

        factorization_set = FactorizationSet(sc_tensor=sc_tensor)

        rank = [rank] if type(rank) == int else rank
        for current_rank in rank:
            print(f"Decomposing tensor of shape {tensor.shape} into rank {current_rank} matrices")

            for i in trange(n_restarts):
                bayesianCP = BayesianCP(
                    dims=tensor.shape, rank=current_rank, init_alpha=init_alpha, init_beta=init_beta, model=model,
                    fixed_mode=fixed_mode, fixed_value=fixed_value, fixed_mode_variance=fixed_mode_variance)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                bayesianCP.to(device)
                tic = time.time()
                bayesianCP.fit(
                    tensor,
                    num_steps=num_steps,
                    initial_lr=initial_lr,
                    lr_decay_gamma=lr_decay_gamma,
                    progress_bar=False
                )
                time_taken = time.time() - tic
                factorization_set.add_precis_factorization(current_rank, i, bayesianCP.precis(tensor))
                params = {'num_steps': num_steps, 'initial_lr': initial_lr, 'init_alpha': init_alpha,
                          'lr_decay_gamma': lr_decay_gamma, 'time_taken': time_taken}
                factorization_set.add_factorization_params(current_rank, i, params)

            if plot_var_explained:
                with plt.rc_context({'figure.figsize': (5, 2)}):
                    var_explained = [factorization_set.variance_explained(rank=current_rank, restart_index=i) for i in
                                     range(n_restarts)]
                    plt.plot(var_explained, 'g')
                    plt.xlabel('Restart')
                    plt.ylabel("Variance Explained")
                    plt.ylim(min(var_explained) - 0.05, 1)
                    plt.show()

        return factorization_set

    @staticmethod
    def factorize_hals(
            sc_tensor: SingleCellTensor,
            rank: Union[int, list[int]],
            n_restarts: int = 10,
            init: str = 'random',
            num_steps: int = 500,
            sparsity_coefficients=None,
            plot_var_explained: bool = True
    ) -> FactorizationSet:
        """ Run CP HALS on the tensor """

        if len(sc_tensor.tensor.shape) > 4:
            print('[Warning] tensor should be 3 or 4 dimensional!')

        tensor = sc_tensor.tensor.astype(float)

        factorization_set = FactorizationSet(sc_tensor=sc_tensor)

        rank = [rank] if type(rank) == int else rank
        for current_rank in rank:
            print(f"Decomposing tensor of shape {tensor.shape} into rank {current_rank} matrices")

            for i in trange(n_restarts):
                tic = time.time()
                hals_factors, errors_hals = non_negative_parafac_hals(
                    tensor,
                    rank=current_rank,
                    init=init,
                    n_iter_max=num_steps,
                    sparsity_coefficients=sparsity_coefficients,
                    return_errors=True
                )
                time_taken = time.time() - tic

                factorization_set.add_mean_factorization(
                    current_rank, i, [{'mean': torch.from_numpy(hals_factors[1][i])} for i in range(len(tensor.shape))]
                )

                params = {'num_steps': num_steps, 'time_taken': time_taken, 'losses': errors_hals}
                factorization_set.add_factorization_params(current_rank, i, params)

            if plot_var_explained:
                with plt.rc_context({'figure.figsize': (5, 2)}):
                    var_explained = [factorization_set.variance_explained(rank=current_rank, restart_index=i) for i in
                                     range(n_restarts)]
                    plt.plot(var_explained, 'g')
                    plt.xlabel('Restart')
                    plt.ylabel("Variance Explained")
                    plt.ylim(min(var_explained) - 0.05, 1)
                    plt.show()

        return factorization_set

    @staticmethod
    def get_consensus_factorization(
            factorization_set: FactorizationSet,
            rank: int,
            model: str = 'zero_inflated_poisson_fixed',
            init_alpha: float = 1e2,
            init_beta: float = 1.,
            fixed_mode: int = 2,
            fixed_mode_variance: float = 10.,
            n_neighbors: int = 50,
            contamination: float = 0.05,
            num_steps: int = 500,
            initial_lr: float = 1e-1,
            lr_decay_gamma: float = 1e-1,
            filter_outliers: bool = False
    ):
        data = np.column_stack([f.gene_factor['mean'].numpy() for f in factorization_set.factorizations[rank].values()]).T
        sums = data.T.sum(axis=0)
        sums[sums == 0] = 1
        data_normed = (data.T / sums).T * 1e5

        kmeans = KMeans(init="k-means++", n_clusters=rank, n_init=20, tol=1e-8)
        labels_ = make_pipeline(StandardScaler(), kmeans).fit(data_normed)[-1].labels_

        if filter_outliers:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outliers = make_pipeline(StandardScaler(), lof).fit_predict(data_normed)
            medians = np.stack(np.median(data_normed[(labels_ == group) & (outliers != -1), :], axis=0)
                               for group in np.unique(labels_))
        else:
            medians = np.stack(np.median(data_normed[(labels_ == group), :], axis=0)
                               for group in np.unique(labels_))

        consensus = SingleCellBTF.factorize(
            sc_tensor=factorization_set.sc_tensor, rank=rank, n_restarts=1, model=model, fixed_mode=fixed_mode,
            fixed_value = torch.from_numpy(medians.T).float(), init_alpha=init_alpha, init_beta=init_beta,
            fixed_mode_variance=fixed_mode_variance, num_steps=num_steps, initial_lr=initial_lr,
            lr_decay_gamma=lr_decay_gamma, plot_var_explained=False,
        )

        gene_factor = consensus.get_factorization(rank=rank, restart_index=0).gene_factor['mean'].numpy()
        mismatch = (1 - np.isclose(medians.T, gene_factor, atol=0.5)).sum()
        var_explained = consensus.variance_explained(rank=rank, restart_index=0).item()
        print(f'{mismatch} / {medians.flatten().shape[0]} mismatches in final gene factors')
        print(f'Variance explained by reconstructed factorization = {var_explained}')
        return consensus
