import torch
import time

import matplotlib.pyplot as plt
from tqdm import trange
from typing import Union

from tensorly.decomposition import non_negative_parafac_hals
from torch.multiprocessing import Pool, set_start_method

from .sc_tensor import SingleCellTensor
from .sc_factors import FactorizationSet
from .bayesian_parafac import BayesianCP

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class SingleCellBTF:
    """ Bayesian Tensor Factorization of single cell rnaseq data """

    @staticmethod
    def factorize(
            sc_tensor: SingleCellTensor,
            rank: Union[int, list[int]],
            model: str = 'gamma_poisson',
            n_restarts: int = 10,
            num_steps: int = 500,
            initial_lr: float = 1,
            init_alpha: int = 1e2,
            fixed_mode: int = None,
            fixed_value=None,
            lr_decay_gamma: float = 1e-3,
            plot_var_explained: bool = True,
            max_parallel_processes: int = 8
    ) -> FactorizationSet:
        """ Run BTF on the tensor """

        if len(sc_tensor.tensor.shape) > 4:
            print('[Warning] tensor should be 3 or 4 dimensional!')

        tensor = torch.from_numpy(sc_tensor.tensor)
        # weights_init, factors_init = initialize_cp(tensor, non_negative=True, init='random', rank=10)

        factorization_set = FactorizationSet(sc_tensor=sc_tensor)

        def factorize_rank(current_rank):
            print(
                f"Decomposing tensor of shape {tensor.shape} into rank {current_rank} matrices using {max_parallel_processes} processes \n")

            pool = Pool(processes=max_parallel_processes)
            results = [
                pool.apply_async(
                    factorize_rank_restart,
                    args=(
                    tensor, current_rank, current_restart, num_steps, initial_lr, lr_decay_gamma, init_alpha, model,
                    fixed_mode, fixed_value)
                ) for current_restart in range(n_restarts)]

            for result in results:
                restart, precis, params = result.get()
                factorization_set.add_precis_factorization(current_rank, restart, precis)
                factorization_set.add_factorization_params(current_rank, restart, params)
            pool.close()

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

        # with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_threads) as executor:
        for res in map(factorize_rank, [rank] if type(rank) == int else rank):
            print(res)

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
                    current_rank, i, [{'mean': torch.from_numpy(hals_factors[1][i])} for i in range(len(tensor.shape))])

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


def factorize_rank_restart(
        tensor, current_rank, current_restart, num_steps, initial_lr, lr_decay_gamma, init_alpha, model, fixed_mode,
        fixed_value
):
    bayesianCP = BayesianCP(dims=tensor.shape, rank=current_rank, init_alpha=init_alpha, model=model,
                            fixed_mode=fixed_mode, fixed_value=fixed_value)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bayesianCP.to(device)
    tic = time.time()
    svi = bayesianCP.fit(
        tensor,
        num_steps=num_steps,
        initial_lr=initial_lr,
        lr_decay_gamma=lr_decay_gamma,
        progress_bar=False
    )
    time_taken = time.time() - tic
    params = {'num_steps': num_steps, 'initial_lr': initial_lr, 'init_alpha': init_alpha,
              'lr_decay_gamma': lr_decay_gamma, 'time_taken': time_taken}
    return current_restart, bayesianCP.precis(tensor), params

