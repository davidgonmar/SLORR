import torch
from torch.optim import Optimizer

from .polar_express import polar_express


def _as_matrix(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 4:
        return weight.reshape(weight.shape[0], -1)
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D or 4-D weight tensor, got {weight.ndim}-D")
    return weight


def _iter_regularized_weights(model, layer_names):
    layer_names = set(layer_names)
    for name, module in model.named_modules():
        if name not in layer_names:
            continue

        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor) or weight.dim() < 2:
            continue
        if not weight.requires_grad:
            raise ValueError(f"Layer {name} weight does not require grad")
        yield name, weight


@torch.no_grad()
def slorr_hoyer_value_grad(weight: torch.Tensor, eps: float = 1e-12, steps: int = 6):
    """Return value and gradient for SLORR-Hoyer."""
    matrix = _as_matrix(weight)
    frob = torch.linalg.norm(matrix, ord="fro") + eps
    polar = polar_express(matrix, steps=steps)
    nuc = torch.sum(matrix * polar)
    reg = nuc / frob
    value = reg**2
    grad = 2 * reg * ((polar / frob) - (nuc / (frob**3)) * matrix)
    return value, grad


@torch.no_grad()
def slorr_nuc_value_grad(weight: torch.Tensor, eps: float = 1e-12, steps: int = 6):
    """Return value and gradient for SLORR-Nuc."""
    del eps
    matrix = _as_matrix(weight)
    polar = polar_express(matrix, steps=steps)
    value = torch.sum(matrix * polar)
    return value, polar


def slorr_hoyer_loss(
    model,
    layer_names,
    eps: float = 1e-12,
    reg_lambda: float = 0.01,
    steps: int = 6,
):
    """
    Add SLORR-Hoyer gradients to selected layer weights.
    """
    if reg_lambda == 0:
        return 0.0

    loss = 0.0
    for _, weight in _iter_regularized_weights(model, layer_names):
        value, grad = slorr_hoyer_value_grad(weight, eps=eps, steps=steps)
        if weight.ndim == 4:
            grad = grad.reshape_as(weight)
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)
        weight.grad.add_(grad, alpha=reg_lambda)
        loss += value
    return loss


def slorr_nuc_loss(
    model,
    layer_names,
    eps: float = 1e-12,
    reg_lambda: float = 0.01,
    steps: int = 6,
):
    """
    Add SLORR-Nuc gradients to selected layer weights.
    """
    if reg_lambda == 0:
        return 0.0

    loss = 0.0
    for _, weight in _iter_regularized_weights(model, layer_names):
        value, grad = slorr_nuc_value_grad(weight, eps=eps, steps=steps)
        if weight.ndim == 4:
            grad = grad.reshape_as(weight)
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)
        weight.grad.add_(grad, alpha=reg_lambda)
        loss += value
    return loss


class AdamSLORRHoyerDecoupled(Optimizer):
    """
    AdamW optimizer with decoupled SLORR-Hoyer update.

    For each weight layer:
      1. Compute the SLORR-Hoyer gradient from the current weight (before any update).
      2. Apply the standard AdamW step (task loss only and wd).
      3. Apply the precomputed SLORR-Hoyer gradient as a separate decoupled update.

    This is similar to Q3R's AdamQ3R in how it applies updates.

    Note: this assumes no weird things like weight tying!

    Args:
        params:           parameter groups (same as AdamW).
        model:            the nn.Module being trained.
        layer_names:      iterable of named_modules keys to regularize.
        lr:               learning rate.
        betas:            Adam beta1, beta2.
        eps:              Adam epsilon.
        weight_decay:     decoupled weight decay coefficient.
        slorr_lambda:     SLORR-Hoyer regularization strength.
        steps:            Polar Express iterations for nuclear norm approximation.
        reg_eps:          small epsilon used inside LowRankRegularizer.
    """

    def __init__(
        self,
        params,
        model,
        layer_names,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        slorr_lambda=0.0,
        steps=6,
        reg_eps=1e-12,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= slorr_lambda:
            raise ValueError(f"Invalid slorr_lambda value: {slorr_lambda}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.model = model
        self.layer_names = set(layer_names)
        self.slorr_lambda = float(slorr_lambda)
        self.steps = int(steps)
        self.reg_eps = float(reg_eps)
        self.last_reg_loss = 0.0

        # _param_to_group: resolve per-param optimizer hyperparameters.
        # _reg_params: precomputed regularized weights as (name, param, group).
        self._param_to_group = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._param_to_group[p] = group

        self._reg_params = []
        for name, module in self.model.named_modules():
            if name not in self.layer_names:
                continue
            weight = getattr(module, "weight", None)
            if weight is None or not isinstance(weight, torch.Tensor) or weight.dim() < 2:
                continue
            group = self._param_to_group.get(weight, None)
            if group is None:
                continue
            self._reg_params.append((name, weight, group))

        self._reg_param_ids = {id(p) for _, p, _ in self._reg_params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_reg_loss = 0.0

        if self.slorr_lambda <= 0.0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._adam_update(p, group)
            return loss

        # Iterate over regularized layers first so we can work with one weight at a time
        # For each such weight: compute SLORR-Hoyer grad (pre-update) -> Adam step -> apply SLORR-Hoyer
        # For all other params: just do the Adam step

        # Note: here even if grad is None, SLORR-Hoyer grad will be computed as long as it is passed
        # to the layer list to regularize.

        # Non-regularized params
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if id(p) in self._reg_param_ids:
                    continue  # handled below
                self._adam_update(p, group)

        # Regularized params
        for name, weight, group in self._reg_params:
            if not weight.requires_grad:
                continue

            # Compute SLORR-Hoyer gradient from current (pre-update) weight
            matrix = weight.reshape(weight.shape[0], -1) if weight.ndim == 4 else weight
            value, grad = slorr_hoyer_value_grad(
                matrix,
                eps=self.reg_eps,
                steps=self.steps,
            )

            self.last_reg_loss += value.item()

            # Standard AdamW update (task loss and wd)
            if weight.grad is not None:
                self._adam_update(weight, group)

            # Decoupled SLORR-Hoyer update
            lr = group["lr"]
            if weight.ndim == 4:
                grad = grad.reshape_as(weight)
            weight.add_(grad, alpha=-lr * self.slorr_lambda)

        return loss

    def _adam_update(self, p, group):
        grad = p.grad
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["mt"] = torch.zeros_like(p)
            state["vt"] = torch.zeros_like(p)

        state["step"] += 1
        t = state["step"]
        mt, vt = state["mt"], state["vt"]
        beta1, beta2 = group["beta1"], group["beta2"]
        lr = group["lr"]
        eps = group["eps"]
        wd = group["weight_decay"]

        mt.mul_(beta1).add_(grad, alpha=1 - beta1)
        vt.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        mt_hat = mt / (1 - beta1**t)
        vt_hat = vt / (1 - beta2**t)
        denom = vt_hat.sqrt().add_(eps)

        if wd != 0:
            p.mul_(1 - lr * wd)

        p.addcdiv_(mt_hat, denom, value=-lr)




# DDP Implementation

def slorr_ddp_reg_loss(
    slorr_params,
    rank,
    world_size,
    slorr_lambda,
    eps,
    steps,
):
    """
    Compute the SLORR-Hoyer regularizer scalar for this rank's partition of layers.

    Each rank processes layers[rank::world_size]. Gradients flow normally through param.grad as DDP's all-reduce propagates them across ranks.

    Returns ``(scaled_reg_total, raw_reg_total)`` where:
    - ``scaled_reg_total`` is what gets added to the training loss, it's multiplied
      by ``world_size`` to cancel DDP's gradient averaging so the effective
      regularization strength is independent of the number of GPUs.
    - ``raw_reg_total`` is the pre-lambda quantity for this rank's layer shard,
      summed across ranks via ``dist.reduce`` in the trainer for logging.
    """
    if slorr_lambda == 0:
        device = slorr_params[0][1].device if slorr_params else torch.device("cpu")
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return zero, zero

    my_params = slorr_params[rank::world_size]

    if not my_params:
        device = slorr_params[0][1].device if slorr_params else torch.device("cpu")
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return zero, zero

    reg_total = None
    raw_reg_total = None
    for name, weight in my_params:
        weight_mat = weight.reshape(weight.shape[0], -1) if weight.ndim == 4 else weight
        weight_fp32 = weight_mat.float()
        value, grad = slorr_hoyer_value_grad(weight_fp32, eps=eps, steps=steps)
        raw_reg = value.detach() + torch.sum(
            (weight_fp32 - weight_fp32.detach()) * grad.detach()
        )
        reg = raw_reg * slorr_lambda
        if reg_total is None:
            reg_total = reg
            raw_reg_total = raw_reg
        else:
            reg_total = reg_total + reg
            raw_reg_total = raw_reg_total + raw_reg

    return reg_total * world_size, raw_reg_total
