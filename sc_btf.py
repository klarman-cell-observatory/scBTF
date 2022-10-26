
from typing import Union

import torch

import matplotlib.pyplot as plt

from tqdm import trange, tqdm

from bayesian_parafac import BayesianCP
from sc_tensor import SingleCellTensor
from sc_factors import FactorizationSet


class SingleCellBTF():
    """ Bayesian Tensor Factorization of single cell rnaseq data """

    @staticmethod
    def factorize(
            sc_tensor: SingleCellTensor,
            rank: Union[int, list[int]],
            n_restarts: int = 10,
            num_steps: int = 500,
            initial_lr: float = 1,
            init_alpha: float = 1e2,
            lr_decay_gamma: float = 1e-3,
            plot_var_explained: bool = True
    ) -> FactorizationSet:
        """ Run BTF on the tensor """

        if len(sc_tensor.tensor.shape) > 4:
            print('[Warning] tensor should be 3 or 4 dimensional!')

        tensor = torch.from_numpy(sc_tensor.tensor)

        factorization_set = FactorizationSet(tensor=sc_tensor.tensor)

        rank = [rank] if type(rank) == int else rank
        for current_rank in rank:
            print(f"Decomposing tensor of shape {tensor.shape} into rank {current_rank} matrices")

            for i in trange(n_restarts):
                bayesianCP = BayesianCP(dims=tensor.shape, rank=current_rank, init_alpha=init_alpha)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                bayesianCP.to(device)

                svi = bayesianCP.fit(
                    tensor,
                    num_steps=num_steps,
                    initial_lr=initial_lr,
                    lr_decay_gamma=lr_decay_gamma
                )
                factorization_set.add_precis_factorization(current_rank, i, bayesianCP.precis(tensor))
                params = {num_steps: num_steps, initial_lr: initial_lr, init_alpha: init_alpha,
                          lr_decay_gamma: lr_decay_gamma}
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
