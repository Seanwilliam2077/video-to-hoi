"""Keep source-authored cylinder proportions during the existing temporal fit.

The bundle still fits one global size, every pose and latent surface attachment.
Projecting the shape gradient onto its uniform-scale subspace prevents a long
foam cylinder from becoming a short thick cylinder to satisfy uncertain depth or
hand constraints. No source/runtime file or learned weight is modified.
"""
import json
from pathlib import Path
import sys

import numpy as np


def constrained_adam(factory, groups, handles, *args, **kwargs):
    """Tie the three shape coordinates by the orthogonal gradient projection."""
    import torch
    shape = groups[2]['params'][0]
    if shape.shape != (3,) or not torch.allclose(shape, shape.mean().expand_as(shape)):
        raise ValueError('Require the original isotropic-initialized bundle shape group')
    handles.append(shape.register_hook(lambda gradient: gradient.mean().expand_as(gradient).clone()))
    return factory(groups, *args, **kwargs)


def main():
    import torch
    from v2hoi import temporal_bundle_runtime as bundle
    from v2hoi.temporal_supported_runtime import main as supported
    from v2hoi.native import sha256
    source = Path(__file__).resolve()
    source_hash = sha256(source)
    original = bundle.fit_surface_bundle

    def fit(*args, **kwargs):
        adam = torch.optim.Adam
        handles = []
        def optimizer(groups, *values, **keywords):
            return constrained_adam(adam, groups, handles, *values, **keywords)
        torch.optim.Adam = optimizer
        try:
            poses, shape, report = original(*args, **kwargs)
        finally:
            torch.optim.Adam = adam
            for handle in handles:
                handle.remove()
        if not np.allclose(shape, np.repeat(shape.mean(), 3), atol=1e-6):
            raise ValueError('The fitted object shape left the uniform-scale subspace')
        report['source_authored_shape_constraint'] = dict(mode='uniform_global_scale',
            preserves_authored_aspect_ratios=True, learned_weights_updated=False,
            pose_and_surface_variables_unchanged=True)
        return poses, shape, report

    bundle.fit_surface_bundle = fit
    try:
        supported()
    finally:
        bundle.fit_surface_bundle = original
    if '--worker' in sys.argv:
        if sha256(source) != source_hash:
            raise ValueError('Uniform-shape constraint adapter changed during inference')
        output = Path(sys.argv[sys.argv.index('--output')+1])
        path = output/'result.json'
        report = json.loads(path.read_text())
        report['inputs'][str(source)] = source_hash
        report['structural_shape_assistance'] = dict(enabled=True, mode='uniform_global_scale',
            scope='Source-authored aspect ratio prior; absolute metric size is still uncertain')
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
