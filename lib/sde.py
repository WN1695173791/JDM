import abc
import numpy as np
import torch
import torch.nn as nn


class AbstractSDE(abc.ABC):
    def __init__(self):
        super().__init__()
        self.N = 1000

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        raise NotImplementedError

    @abc.abstractmethod
    def sde(self, x_t, y_t, t):
        """Compute the drift/diffusion of the forward SDE
        dx = b(x_t, y_t, t)dt + s(x_t, y_t, t)dW
        """
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob(self, x_0, y_0, t):
        """Compute the mean/std of the transitional kernel
        p(x_t, y_t | x_0, y_0).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def scale_start_to_noise(self, t):
        """Compute the scale of conversion
        from the original image estimation loss, i.e, || x_0 - x_0_pred ||
        to the noise prediction loss, i.e, || e - e_pred ||.
        Denoting the output of this function by C, 
        C * || x_0 - x_0_pred || = || e - e_pred || holds.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def proposal_distribution(self):
    #     raise NotImplementedError

    def reverse(self, model, model_pred_type='noise'):
        """The reverse-time SDE."""
        sde_fn = self.sde
        marginal_fn = self.marginal_prob

        class RSDE(self.__class__):
            def __init__(self):
                pass
                
            def sde(self, x_t, y_t, t):
                # Get score function values
                if model_pred_type == 'noise':
                    x_noise_pred, y_noise_pred = model(x_t, y_t, t)
                    _, _, x_std, y_std = marginal_fn(
                        torch.zeros_like(x_t),
                        torch.zeros_like(y_t),
                        t,
                    ) 
                    score_x = -x_noise_pred / x_std
                    score_y = -y_noise_pred / y_std

                elif model_pred_type == 'original':
                    x_0_pred, y_0_pred = model(x_t, y_t, t)
                    x_mean, y_mean, x_std, y_std = marginal_fn(
                        x_0_pred,
                        y_0_pred,
                        t,
                    )
                    score_x = (x_mean - x_t) / x_std
                    score_y = (y_mean - y_t) / y_std

                # Forward SDE's drift & diffusion
                drift_x, drift_y, diffusion_x, diffusion_y = sde_fn(x_t, y_t, t)

                # Reverse SDE's drift & diffusion (Anderson, 1982)
                drift_x = drift_x - diffusion_x ** 2 * score_x
                drift_y = drift_y - diffusion_y ** 2 * score_y
                return drift_x, drift_y, diffusion_x, diffusion_y

        return RSDE()


class VPSDE(AbstractSDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        # self.IS_dist, self.norm_const = self.proposal_distribution()

    @property
    def T(self):
        return 1

    def sde(self, x_t, y_t, t):
        beta_t = (self.beta_0 + t * (self.beta_1 - self.beta_0))[:, None, None, None]
        drift_x = -0.5 * beta_t * x_t
        drift_y = -0.5 * beta_t * y_t
        diffusion = torch.sqrt(beta_t)
        return drift_x, drift_y, diffusion, diffusion

    def marginal_prob(self, x_0, y_0, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )[:, None, None, None]
        marginal_mean_x = torch.exp(log_mean_coeff) * x_0
        marginal_mean_y = torch.exp(log_mean_coeff) * y_0
        marginal_std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return marginal_mean_x, marginal_mean_y, marginal_std, marginal_std

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = - N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def scale_start_to_noise(self, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )[:, None, None, None]
        marginal_coeff = torch.exp(log_mean_coeff)
        marginal_std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        x_scale = marginal_coeff / (marginal_std + 1e-12)
        y_scale = marginal_coeff / (marginal_std + 1e-12)
        return x_scale, y_scale

    # def proposal_distribution(self):
    #     def g2(t):
    #         return self.beta_0 + t * (self.beta_1 - self.beta_0)
    #     def a2(t):
    #         log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) \
    #             - 0.5 * t * self.beta_0
    #         return 1. - torch.exp(2. * log_mean_coeff)
    #     t = torch.arange(1, 1001) / 1000
    #     p = g2(t) / a2(t)
    #     normalizing_const = p.sum()
    #     return p, normalizing_const



