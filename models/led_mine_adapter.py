"""Expose local architectures to LED while preserving each repository's RAW order."""
from copy import deepcopy
import torch
from torch import nn


class CurrentRepoDenoiser(nn.Module):
    def __init__(self, architecture, settings=None):
        super().__init__()
        settings = dict(settings or {})
        if architecture == 'repncb':
            from learning_wt.learning_dwt import learning_dwt_kwargs
            from models.learning_dwt_repncb import LearningDWTRepNCB, refiner_kwargs
            self.network = LearningDWTRepNCB(learning_dwt_kwargs(settings),
                                            refine_config=refiner_kwargs(settings))
        elif architecture == 'mrlfn':
            from models.mrlfn_arch import MRLFN
            self.network = MRLFN(**settings)
        elif architecture == 'splitternet':
            from models.paper_denoisers import build_paper_denoiser
            if settings:
                raise ValueError('SplitterNet uses its fixed local architecture')
            self.network = build_paper_denoiser('splitternet')
        elif architecture == 'nafnet':
            from models.natnet_arch import NAFNet
            self.network = NAFNet(**settings)
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        # LED [R,G1,B,G2] <-> local [R,G1,G2,B]; self-inverse.
        self.register_buffer('channel_order', torch.tensor([0, 1, 3, 2]), persistent=False)

    def forward(self, x):
        return self.network(x.index_select(1, self.channel_order)).index_select(1, self.channel_order)

    def deploy(self):
        result = deepcopy(self).eval()
        if callable(getattr(result.network, 'deploy', None)):
            result.network = result.network.deploy()
        return result
