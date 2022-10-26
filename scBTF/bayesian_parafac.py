from tqdm import trange
import warnings

import torch
import torch.nn as nn
import torch.distributions.constraints as constraints

import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive

assert pyro.__version__.startswith('1.8.2')
# clear the param store in case we're in a REPL
pyro.clear_param_store()

warnings.simplefilter(action='ignore', category=FutureWarning)

seed = 20221010
torch.manual_seed(seed)
pyro.set_rng_seed(seed)


class BayesianCP(nn.Module):
    """
    Bayesian Canonical Polyadic decomposition via Variational Inference.
    
    Parameters
    ----------
    rank
        
    dims
    
    init_alpha
    
    Examples
    --------
    >>> true_rank = 3
    >>> I, J, K = 30, 40, 20
    >>> A_factor_matrix = torch.randint(10, size=(I, true_rank))
    >>> B_factor_matrix = torch.randint(10, size=(J, true_rank))
    >>> C_factor_matrix = torch.randint(10, size=(K, true_rank))
    >>> tensor = torch.einsum('ir,jr,kr->ijk', A_factor_matrix, B_factor_matrix, C_factor_matrix)
    >>> 
    >>> bayesianCP = BayesianCP(dims=tensor.shape, rank=3, init_alpha=1)
    >>> bayesianCP.to(device)
    >>> svi = bayesianCP.fit(tensor, num_steps = 1200)
    >>> prs = bayesianCP.precis(tensor)
    >>> a, b, c = [prs[f'factor_{i}']['mean'] for i in range(3)]

    >>> tensor_means = torch.einsum('ir,jr,kr->ijk', a, b, c)
    >>> tensor_sampled = pyro.distributions.Poisson(torch.einsum('ir,jr,kr->ijk', a, b, c)).sample()

    >>> print(1 - (torch.norm(tensor.float() - tensor_means.float())**2 / torch.norm(tensor.float())**2))
    >>> print(1 - (torch.norm(tensor.float() - tensor_sampled.float())**2 / torch.norm(tensor.float())**2))
    >>> 
    
    Notes
    -----
    See further usage examples in the following tutorials:
    1. :doc:`/tutorials/notebooks/api_overview`
    """

    def __init__(
            self,
            dims,
            rank: int = 3,
            init_alpha: float = 0.1,
            model: str = 'gamma_poisson'
    ):
        super().__init__()

        self.rank = rank
        self.dims = dims
        self.init_alpha = init_alpha

        models = {
            'gamma_poisson': [self.model_gamma_poisson, self.guide_gamma_poisson],
            'truncated_gaussian': []
        }

        self.model, self.guide = models[model]

    def model_gamma_poisson(self, data):
        """ Gamma Poisson model """
        factor = []
        for mode in range(len(self.dims)):
            alpha = torch.ones([self.dims[mode], self.rank])
            beta = torch.ones([self.dims[mode], self.rank])
            factor.append(pyro.sample(f"factor_{mode}", pyro.distributions.Gamma(alpha, beta).to_event()))

        rate = torch.einsum(BayesianCP.get_einsum_formula(len(self.dims)), *factor)
        pyro.sample("obs", pyro.distributions.Poisson(rate).to_event(), obs=data)

    def guide_gamma_poisson(self, data):
        """ Guide for Gamma Poisson model """
        for mode in range(len(self.dims)):
            init_alpha = torch.ones([self.dims[mode], self.rank]) * self.init_alpha
            init_beta = torch.ones([self.dims[mode], self.rank])

            q_alpha = pyro.param(f"Q{mode}_alpha", init_alpha, constraint=constraints.greater_than(1e-5))
            q_beta = pyro.param(f"Q{mode}_beta", init_beta, constraint=constraints.positive)

            pyro.sample(f"factor_{mode}", pyro.distributions.Gamma(q_alpha, q_beta).to_event())

    def model_truncated_gaussian(self, data, sigma_init_high=10.):
        """ Truncated Gaussian model """
        factor = []
        for mode in range(len(self.dims)):
            factor_loc = torch.ones([self.dims[mode], self.rank])
            factor_scale = torch.ones([self.dims[mode], self.rank])

            factor.append(
                pyro.sample(f"factor_{mode}", pyro.distributions.HalfNormal(factor_loc, factor_scale).to_event()))

        loc = torch.einsum(BayesianCP.get_einsum_formula(len(self.dims)), *factor)
        sigma = pyro.sample("sigma", pyro.distributions.Uniform(torch.zeros_like(loc),
                                                                torch.ones_like(loc) * sigma_init_high).to_event())
        pyro.sample("obs", pyro.distributions.HalfNormal(loc, sigma).to_event(), obs=data)

    def guide_truncated_gaussian(self, data, sigma_init_high=10.):
        """ Guide for Truncated Gaussian model """
        for mode in range(len(self.dims)):
            q_factor_loc = pyro.param(f"Q{mode}_loc", torch.ones([self.dims[mode], self.rank]),
                                      constraint=constraints.positive)
            q_factor_scale = pyro.param(f"Q{mode}_scale", torch.ones([self.dims[mode], self.rank]),
                                        constraint=constraints.positive)

            pyro.sample(f"factor_{mode}", pyro.distributions.HalfNormal(q_factor_loc, q_factor_scale).to_event())

        q_sigma_low = pyro.param(f"Q_sigma_low", torch.zeros_like(data), constraint=constraints.positive)
        q_sigma_high = pyro.param(f"Q_sigma_high", torch.ones_like(data) * sigma_init_high,
                                  constraint=constraints.positive)
        pyro.sample("sigma", pyro.distributions.Uniform(q_sigma_low, q_sigma_high).to_event())

    def fit(self, data, num_steps=3000, initial_lr=0.01, lr_decay_gamma=0.1):
        pyro.clear_param_store()

        optimizer = pyro.optim.ClippedAdam({
            'lr': initial_lr,
            'lrd': lr_decay_gamma ** (1 / num_steps),
            'betas': (0.90, 0.999)}
        )

        # reparam_model = AutoReparam()(model)
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        self.losses = []
        bar = trange(num_steps)
        for step in bar:
            loss = svi.step(data)
            self.losses.append(loss)
            if step % 100 == 0:
                bar.set_postfix(loss='{:.2e}'.format(loss))
        return svi

    def posterior_predictive_samples(self, data, num_samples=100, per_component=False):
        """ Sample from the posterior predictive distribution """
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        if per_component:
            samples = predictive(data)
            node_supports = {k: v for k, v in samples.items() if k != "obs"}
            return {f'c_{i}': {k: v[:, :, i] for k, v in node_supports.items()} for i in range(self.rank)}
        else:
            return predictive(data)

    def precis(self, data, num_samples=100):
        """ Get the mean standard deviation and high density interval of the posterior distribution """
        precis_factors = {}

        predictive = self.posterior_predictive_samples(data, num_samples)
        node_supports = {k: v for k, v in predictive.items() if k != "obs"}

        for node, support in node_supports.items():
            hpdi = pyro.ops.stats.hpdi(support, prob=0.89, dim=0)
            precis_factors[node] = {
                'mean': support.mean(0),
                'std': support.std(0),
                '|0.89': hpdi[0],
                '0.89|': hpdi[1]
            }
        return precis_factors

    @staticmethod
    def get_einsum_formula(num_modes):
        einsum_formula = {
            2: 'ir,jr->ij',
            3: 'ir,jr,kr->ijk',
            4: 'ir,jr,kr,lr->ijkl'
        }
        return einsum_formula[num_modes]
