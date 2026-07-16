# SLORR

This repository contains code for **SLORR: Simple and Efficient In-Training Low-Rank Regularization** by David González-Martínez and Shiwei Liu.

Paper: https://arxiv.org/abs/2607.08754

## Overview

SLORR is a stateless, architecture-preserving framework for in-training low-rank regularization. This initial release includes:

- `SLORR-Hoyer`: the Hoyer-style regularizer.
- `SLORR-Nuc`: the nuclear-norm regularizer.
- A decoupled AdamW optimizer for `SLORR-Hoyer`.
- A DDP-oriented helper implementation for LLM-style training loops.
- A basic ImageNet example script using `timm`.

Exact reproduction scripts for the paper will be released in the near future.

## Files

- `slorr/polar_express.py`: Polar Express approximation used by the regularizers.
- `slorr/regularizers.py`: `SLORR-Hoyer`, `SLORR-Nuc`, decoupled AdamW, and DDP helper functions.
- `examples/basic_example.py`: ImageNet training example.

## Basic Usage

```python
import torch
import torch.nn.functional as F

from slorr import slorr_hoyer_loss

optimizer.zero_grad(set_to_none=True)

logits = model(x)
loss = F.cross_entropy(logits, y)

# adds grads to the layers inside
regloss = slorr_hoyer_loss(
    model,
    layer_names=reg_layers,
    reg_lambda=1e-1,
    steps=6,
)

loss.backward()
optimizer.step()
```

Use `slorr_nuc_loss` for `SLORR-Nuc`.

## Decoupled AdamW

```python
from slorr import AdamSLORRHoyerDecoupled

optimizer = AdamSLORRHoyerDecoupled(
    model.parameters(),
    model=model,
    layer_names=reg_layers,
    lr=1e-2,
    weight_decay=1e-4,
    slorr_lambda=1e-1,
    steps=6,
)
```

## ImageNet Example

The example expects an ImageNet-style directory with `train/` and `val/` subdirectories.

```bash
python examples/basic_example.py --data /path/to/imagenet --method slorr_hoyer
python examples/basic_example.py --data /path/to/imagenet --method slorr_nuc
python examples/basic_example.py --data /path/to/imagenet --method slorr_hoyer_decoupled
```

## Citation

```bibtex
@misc{gonzalezmartinez2026slorr,
  title={SLORR: Simple and Efficient In-Training Low-Rank Regularization},
  author={David Gonz{\'a}lez-Mart{\'i}nez and Shiwei Liu},
  year={2026},
  eprint={2607.08754},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2607.08754}
}
```

## License

This code is released under the MIT License. See `LICENSE`.
