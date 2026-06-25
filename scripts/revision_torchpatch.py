"""Compatibility shim: make torch.load default to weights_only=False.

torch >= 2.6 flipped the torch.load `weights_only` default to True, which rejects
chemprop 1.6.1 checkpoints (they pickle an argparse.Namespace / TrainArgs). These
are our own trusted checkpoints, so we restore the pre-2.6 behaviour. Import this
module once before any chemprop.utils.load_checkpoint call.
"""
import functools
import torch

if not getattr(torch.load, "_wo_patched", False):
    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._wo_patched = True
    torch.load = _load
