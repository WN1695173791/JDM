from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from .sde import AbstractSDE


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        sde: AbstractSDE,
        model,
        model_pred_type: str = 'noise',
        t_eps: float = 1e-5,
    ):
        """Constructor.
        
        Args:
            sde: Reference SDE for x, y.
            model: Score function model.
            model_pred_type: Type of the outputs of the model.
                If 'noise', the model predicts noise (eps_x, eps_y).
                If 'original', the model predicts the original inputs (x_0, y_0).
            t_eps: Start-time in SDE.
                Defaults to 1e-5.
        """
        super().__init__()
        self.sde = sde
        self.model = model
        self.rsde = sde.reverse(model)
        assert model_pred_type in ['noise', 'original']
        self.model_pred_type = model_pred_type
        self.t_eps = t_eps

    @property
    def device(self):
        return next(self.parameters()).device

    def predictor(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        no_noise: bool = False,
    ):
        # Euler-Maruyama
        dt = 1. / self.sde.N
        z1 = torch.randn_like(x_t)
        z2 = torch.randn_like(y_t)
        drift_x, drift_y, diffusion_x, diffusion_y = self.rsde.sde(x_t, y_t, t)
        x_prev = x_t - drift_x * dt
        y_prev = y_t - drift_y * dt
        if not no_noise:
            x_prev += diffusion_x * np.sqrt(dt) * z1
            y_prev += diffusion_y * np.sqrt(dt) * z2
        return x_prev, y_prev

    def corrector(self, time_step, x_t, y_t):
        # Empty yet! 
        return x_t, y_t

    def forward(
        self,
        x_T: torch.Tensor,
        y_T: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        y_0: Optional[torch.Tensor] = None,
        option: str = 'joint',
        pbar: Optional[tqdm] = None,
    ):
        if option == 'joint':
            assert x_0 is None and y_0 is None
        elif option == 'x->y':
            assert x_0 is not None and y_0 is None
        elif option == 'y->x':
            assert y_0 is not None and x_0 is None
        else:
            raise ValueError(
                f'Option {option} should be one of "joint", "x->y" and "y->x".'
            )

        B = x_T.shape[0]
        device = x_T.device
        x_t = x_T
        y_t = y_T
        timesteps = torch.linspace(self.t_eps, self.sde.T, self.sde.N)

        for time_step in reversed(range(self.sde.N)):

            # Set continuous time t (in [0, 1])
            t = timesteps[time_step]
            t = t * torch.ones(B).to(device)

            # Last timestep -> no noise (only drift term is applied)
            if time_step > 0:
                no_noise = False
            else:
                no_noise = True

            # Predictor
            x_t, y_t = self.predictor(t, x_t, y_t, no_noise)

            # Corrector
            x_t, y_t = self.corrector(t, x_t, y_t)

            # x->y (controllable generation)
            if x_0 is not None:
                x_mean, _, x_std, _ = self.sde.marginal_prob(x_0, torch.zeros_like(y_t), t)
                x_t = x_mean + x_std * torch.randn_like(x_t)

            # y->x (controllable generation)
            elif y_0 is not None:
                _, y_mean, _, y_std = self.sde.marginal_prob(torch.zeros_like(x_t), y_0, t)
                y_t = y_mean + y_std * torch.randn_like(y_t)

            # (Optional) show the current time step via tqdm
            if pbar is not None:
                pbar.set_postfix_str(f'{option} {time_step}')

        x_0 = x_t
        y_0 = y_t
        return torch.clip(x_0, -1., 1.), torch.clip(y_0, -1., 1.)


