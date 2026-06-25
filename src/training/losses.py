"""Custom loss functions for multi-task ionic liquid property prediction."""

import torch
import torch.nn as nn
import math


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns a log-variance parameter per task that automatically balances
    task contributions. Tasks with higher noise get down-weighted.
    Loss per task: (1 / 2*sigma^2) * MSE + log(sigma)
    """

    def __init__(self, target_names: list = None):
        super().__init__()
        self.target_names = target_names or [
            "gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"
        ]
        num_tasks = len(self.target_names)
        # Learnable log-variance per task (initialized to 0 => sigma=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        per_task_mse = torch.mean((predictions - targets) ** 2, dim=0)

        # precision = 1/(2*sigma^2), regularizer = log(sigma) = 0.5*log_var
        precision = torch.exp(-self.log_vars)
        weighted = precision * per_task_mse + self.log_vars
        total = weighted.sum()

        losses = {"total": total}
        for i, name in enumerate(self.target_names):
            losses[name] = per_task_mse[i]
        # Expose learned weights for logging
        losses["learned_weights"] = torch.exp(-self.log_vars).detach()
        return losses


class PhysicsInformedLoss(nn.Module):
    """Physics-informed multi-task loss with thermodynamic consistency penalties.

    Combines MSE (or uncertainty-weighted) loss with soft constraints:
    1. Gibbs-Helmholtz: G_E ≈ H_E - T*S_E (approximated via correlation)
    2. Gamma-G_E consistency: G_E relates to ln(gamma) via RT*x*ln(gamma)
    3. Clausius-Clapeyron: d(ln P)/d(1/T) ∝ -H_vap (P and H_vap anti-correlate)

    Since targets are standardized, we enforce relative consistency constraints
    (correlations and monotonicity) rather than exact physical equations.
    """

    def __init__(
        self,
        task_weights: dict = None,
        target_names: list = None,
        use_uncertainty: bool = True,
        physics_weight: float = 0.1,
    ):
        super().__init__()
        self.target_names = target_names or [
            "gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"
        ]
        self.physics_weight = physics_weight
        self.use_uncertainty = use_uncertainty

        if use_uncertainty:
            self.base_loss = UncertaintyWeightedLoss(self.target_names)
        else:
            self.base_loss = MultiTaskMSELoss(task_weights, self.target_names)

        # Target indices for physics constraints
        self._idx = {name: i for i, name in enumerate(self.target_names)}

    def _gibbs_helmholtz_penalty(self, pred: torch.Tensor) -> torch.Tensor:
        """G_E and H_E should be positively correlated (G_E ≈ H_E - TS_E).
        Penalize when their predicted directions disagree within a batch."""
        g_e = pred[:, self._idx["G_E"]]
        h_e = pred[:, self._idx["H_E"]]
        # Correlation-based: penalize negative correlation
        g_centered = g_e - g_e.mean()
        h_centered = h_e - h_e.mean()
        cov = (g_centered * h_centered).mean()
        g_std = g_centered.std().clamp(min=1e-6)
        h_std = h_centered.std().clamp(min=1e-6)
        corr = cov / (g_std * h_std)
        # Penalize when correlation < 0.5 (they should be strongly correlated)
        return torch.relu(0.5 - corr)

    def _gamma_ge_penalty(self, pred: torch.Tensor) -> torch.Tensor:
        """G_E relates to activity coefficients. gamma1, gamma2, G_E, G_mix
        should all be positively correlated in the normalized space."""
        g_e = pred[:, self._idx["G_E"]]
        g_mix = pred[:, self._idx["G_mix"]]
        g_centered = g_e - g_e.mean()
        m_centered = g_mix - g_mix.mean()
        cov = (g_centered * m_centered).mean()
        g_std = g_centered.std().clamp(min=1e-6)
        m_std = m_centered.std().clamp(min=1e-6)
        corr = cov / (g_std * m_std)
        return torch.relu(0.5 - corr)

    def _clausius_clapeyron_penalty(self, pred: torch.Tensor) -> torch.Tensor:
        """H_vap and P should be anti-correlated (higher H_vap => lower P at same T).
        Penalize positive correlation."""
        h_vap = pred[:, self._idx["H_vap"]]
        p = pred[:, self._idx["P"]]
        h_centered = h_vap - h_vap.mean()
        p_centered = p - p.mean()
        cov = (h_centered * p_centered).mean()
        h_std = h_centered.std().clamp(min=1e-6)
        p_std = p_centered.std().clamp(min=1e-6)
        corr = cov / (h_std * p_std)
        # They should be negatively correlated; penalize if corr > -0.2
        return torch.relu(corr + 0.2)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        losses = self.base_loss(predictions, targets)

        # Physics penalties (only meaningful with batch_size > 1)
        if predictions.shape[0] > 2:
            gh_penalty = self._gibbs_helmholtz_penalty(predictions)
            gamma_penalty = self._gamma_ge_penalty(predictions)
            cc_penalty = self._clausius_clapeyron_penalty(predictions)
            physics_loss = gh_penalty + gamma_penalty + cc_penalty
            losses["physics"] = physics_loss
            losses["total"] = losses["total"] + self.physics_weight * physics_loss

        return losses


class MultiTaskMSELoss(nn.Module):
    """Weighted multi-task MSE loss.

    Each target property can have a different weight to balance their
    contributions to the total loss.
    """

    def __init__(self, task_weights: dict = None, target_names: list = None):
        super().__init__()
        self.target_names = target_names or [
            "gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"
        ]
        weights = []
        for name in self.target_names:
            w = (task_weights or {}).get(name, 1.0)
            weights.append(w)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute weighted multi-task MSE loss.

        Args:
            predictions: (B, num_targets)
            targets: (B, num_targets)

        Returns:
            dict with 'total' loss and per-task losses
        """
        per_task_mse = torch.mean((predictions - targets) ** 2, dim=0)  # (num_targets,)
        weighted = per_task_mse * self.weights
        total = weighted.sum()

        losses = {"total": total}
        for i, name in enumerate(self.target_names):
            losses[name] = per_task_mse[i]

        return losses


class MultiTaskHuberLoss(nn.Module):
    """Weighted multi-task Huber (smooth L1) loss. More robust to outliers."""

    def __init__(self, task_weights: dict = None, target_names: list = None, delta: float = 1.0):
        super().__init__()
        self.target_names = target_names or [
            "gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"
        ]
        self.delta = delta
        weights = []
        for name in self.target_names:
            w = (task_weights or {}).get(name, 1.0)
            weights.append(w)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        huber = torch.where(
            abs_diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta),
        )
        per_task = huber.mean(dim=0)
        weighted = per_task * self.weights
        total = weighted.sum()

        losses = {"total": total}
        for i, name in enumerate(self.target_names):
            losses[name] = per_task[i]
        return losses
