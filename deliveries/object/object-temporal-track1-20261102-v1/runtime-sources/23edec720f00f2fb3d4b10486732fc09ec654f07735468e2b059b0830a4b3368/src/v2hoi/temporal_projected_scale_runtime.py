"""Avoid treating a flat source object as an upright canonical-height ruler.

This explicit foam-roll variant retains the existing angular-extent/depth size
estimate and uniform shape constraint. Source depth remains an uncertain prior;
no object pose, reference dimension or camera calibration is authored here.
"""
import json
from pathlib import Path
import sys


def projected_scale(depth, mask, intrinsics, extents, normal=None, offset=None):
    from v2hoi.temporal_object import initialize_scale
    scale, report = initialize_scale(depth, mask, intrinsics, extents)
    report.update(canonical_upright_height_prior_disabled=True,
        reason='Canonical Z height does not measure the observed height of a flat object')
    return scale, report


def main():
    from v2hoi import temporal_object_runtime as runtime
    from v2hoi.temporal_uniform_shape_runtime import main as uniform
    from v2hoi.native import sha256
    source = Path(__file__).resolve()
    source_hash = sha256(source)
    original = runtime.initialize_scale
    runtime.initialize_scale = projected_scale
    try:
        uniform()
    finally:
        runtime.initialize_scale = original
    if '--worker' in sys.argv:
        if sha256(source) != source_hash:
            raise ValueError('Projected-size initializer changed during inference')
        output = Path(sys.argv[sys.argv.index('--output')+1])
        path = output/'result.json'
        report = json.loads(path.read_text())
        report['inputs'][str(source)] = source_hash
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
