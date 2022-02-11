import torch
import torch.nn as nn

from .sde import AbstractSDE


class DiffusionTrainer(nn.Module):
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
        assert model_pred_type in ['noise', 'original']
        self.model_pred_type = model_pred_type
        self.t_eps = t_eps
            
    def forward(
        self,
        x_0: torch.Tensor,
        y_0: torch.Tensor,
    ):
        B = x_0.shape[0]
        device = x_0.device

        # Forward process
        t = torch.rand(B, device=device) * (self.sde.T - self.t_eps) + self.t_eps
        x_noise, y_noise = torch.randn_like(x_0), torch.randn_like(y_0)
        x_mean, y_mean, x_std, y_std = self.sde.marginal_prob(x_0, y_0, t)
        x_t = x_mean + x_std * x_noise
        y_t = y_mean + y_std * y_noise

        if self.model_pred_type == 'noise':
            x_noise_pred, y_noise_pred = self.model(x_t, y_t, t)
            # x loss
            x_loss = torch.square(x_noise_pred - x_noise)
            x_loss = x_loss.reshape(B, -1)
            # y loss
            y_loss = torch.square(y_noise_pred - y_noise)
            y_loss = y_loss.reshape(B, -1)

        elif self.model_pred_type == 'original':
            x_0_pred, y_0_pred = self.model(x_t, y_t, t)
            x_scale, y_scale = self.sde.scale_start_to_noise(t)
            x_scale = torch.clip(x_scale, min=None, max=5.0)
            y_scale = torch.clip(y_scale, min=None, max=5.0)
            # x loss
            x_loss = torch.square((x_0_pred - x_0) * x_scale)
            x_loss = x_loss.reshape(B, -1)
            # y loss
            y_loss = torch.square((y_0_pred - y_0) * y_scale)
            y_loss = y_loss.reshape(B, -1)

        return x_loss, y_loss

        
