# NOT WORKING YET
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Sequence, Tuple, Dict, List, Union
from dataclasses import dataclass, field
from .FitAPI import Fitter
from .searcher import Searcher

@dataclass
class TorchFitResult:
    """Lightweight container mimicking the parts of Minuit you usually use."""
    params: Dict[str, float]
    loss: float
    history: List[float] = field(repr=False)     # full loss curve (optional)


class TorchFitter(Fitter):
    """
    Gradient-based fitter with *soft* inequality/equality constraints.

    Parameters
    ----------
    All positional/keyword arguments of :class:`Fitter`, plus

    constraints : list[tuple[str, str, str]], optional
        Each entry is (lhs, op, rhs) where op is one of ">", ">=", "<", "<=", "==".
        Both lhs and rhs must be parameter names.
    penalty_weight : float, default 1e3
        Multiplier for every constraint penalty term.
    epochs : int, default 5000
    optimizer_cls : torch.optim.Optimizer subclass, default torch.optim.Adam
    lr : float, default 1e-3
    device, dtype : forwarded to torch.tensor(...)
    """

    def __init__(self,
                 *args,
                 constraints: Optional[Sequence[Tuple[str, str, str]]] = None,
                 penalty_weight: float = 1e3,
                 epochs: int = 5000,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 lr: float = 1e-3,
                 device: Union[str, torch.device, None] = None,
                 dtype=torch.float32,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.constraints = constraints or []
        self.penalty_weight = float(penalty_weight)
        self.epochs = int(epochs)
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.device = torch.device(device or "cpu")
        self.dtype = dtype

        # -- build torch parameters -------------------------------------------------
        active = [n for n, active in self.model.active_params.items() if active]
        self._torch_params = nn.ParameterDict({
            n: nn.Parameter(torch.tensor(float(getattr(self.model, n)),
                                         dtype=self.dtype, device=self.device))
            for n in active
        })
        # store constants for fixed parameters
        self._fixed = {n: torch.tensor(float(getattr(self.model, n)),
                                       dtype=self.dtype, device=self.device)
                       for n in self.model.param_names if n not in active}

        # data tensors (once)
        self._x  = torch.tensor(self.x,  dtype=self.dtype, device=self.device)
        self._y  = torch.tensor(self.y,  dtype=self.dtype, device=self.device)
        self._ye = torch.tensor(
            np.maximum(1e-5, self.y if self.yerr is None else self.yerr),
            dtype=self.dtype, device=self.device)

    # -------------------------------------------------------------------------
    def _current_param_dict(self) -> Dict[str, torch.Tensor]:
        """Combine learnable and fixed params into one dict of tensors."""
        out = {**self._fixed}
        out.update({k: p for k, p in self._torch_params.items()})
        return out

    def _loss(self) -> torch.Tensor:
        """Least-squares loss + soft penalties."""
        p = self._current_param_dict()
        y_pred = self.model.model_function(self._x, **p)   # MUST be torch ops
        residual = (y_pred - self._y) / self._ye
        loss = torch.mean(residual ** 2)

        # ---- bounds penalties ----------------------------------------------------
        for name, (lo, hi) in self.model.param_bounds.items():
            param = p[name]
            if not np.isneginf(lo):
                loss = loss + self.penalty_weight * torch.relu(lo - param) ** 2
            if not np.isposinf(hi):
                loss = loss + self.penalty_weight * torch.relu(param - hi) ** 2

        # ---- relative constraints penalties -------------------------------------
        eps = 0.0  # no margin; change if you want ≥ instead of >
        for a, op, b in self.constraints:
            pa, pb = p[a], p[b]
            if op == ">":
                loss = loss + self.penalty_weight * torch.relu(pb - pa + eps) ** 2
            elif op == ">=":
                loss = loss + self.penalty_weight * torch.relu(pb - pa) ** 2
            elif op == "<":
                loss = loss + self.penalty_weight * torch.relu(pa - pb + eps) ** 2
            elif op == "<=":
                loss = loss + self.penalty_weight * torch.relu(pa - pb) ** 2
            elif op in {"==", "="}:
                loss = loss + self.penalty_weight * (pa - pb) ** 2
            else:
                raise ValueError(f"Unknown operator {op!r} in constraints")
        return loss

    # -------------------------------------------------------------------------
    def fit(self, searcher: Optional[Searcher] = None) -> TorchFitResult:
        # replicate Searcher integration from the base class
        if searcher is not None:
            if not isinstance(searcher, Searcher):
                raise ValueError("searcher must be a Searcher instance")
            searcher.search(self.data, self.model)
            self.model.assing_params(searcher.params)
            for n, p in self._torch_params.items():
                p.data.fill_(getattr(self.model, n))

        # rebuild optimizer (after possible new initial guess)
        opt = self.optimizer_cls(self._torch_params.values(), lr=self.lr)
        history: List[float] = []

        for epoch in range(self.epochs):
            opt.zero_grad()
            loss = self._loss()
            loss.backward()
            opt.step()
            history.append(float(loss))

        # fill back the converged values
        for n, p in self._torch_params.items():
            setattr(self.model, n, float(p.detach().cpu()))
        # (fixed params were never changed)

        result = TorchFitResult(
            params={n: float(getattr(self.model, n)) for n in self.model.param_names},
            loss=float(history[-1]),
            history=history
        )
        return result
