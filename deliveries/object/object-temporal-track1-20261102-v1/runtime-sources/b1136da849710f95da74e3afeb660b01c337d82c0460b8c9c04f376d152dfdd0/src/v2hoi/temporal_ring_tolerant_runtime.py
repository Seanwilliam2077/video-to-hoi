"""Thin-ring candidate rasterization with one-pixel mask uncertainty.

Only candidate admission/rendering accounts for the quantized silhouette boundary.
The final bundle objective and audit use the original unexpanded renderer. This
is an explicit initializer variant, not evidence of equivalence to EP15.
"""
import json
from pathlib import Path
import sys


def main():
    import torch.nn.functional as functional
    from v2hoi import temporal_ring_runtime as rings
    from v2hoi import temporal_object_runtime as runtime
    from v2hoi.object_refine_runtime import ObservationRenderer
    from v2hoi.native import sha256
    original=runtime.keyframe_candidates
    original_render=ObservationRenderer.render
    own_hash=sha256(Path(__file__))
    def render(*args,**keywords):
        rgb,alpha,depth=original_render(*args,**keywords)
        expanded=functional.max_pool2d(alpha.permute(0,3,1,2),3,stride=1,padding=1).permute(0,2,3,1)
        return rgb,expanded,depth
    def candidates(*args,**keywords):
        ObservationRenderer.render=render
        try:
            return original(*args,**keywords)
        finally:
            ObservationRenderer.render=original_render
    runtime.keyframe_candidates=candidates
    rings.main()
    if '--worker' in sys.argv:
        output=Path(sys.argv[sys.argv.index('--output')+1])
        if sha256(Path(__file__))!=own_hash:
            raise ValueError('Thin-ring candidate uncertainty adapter changed')
        path=output/'result.json'
        report=json.loads(path.read_text())
        report['inputs'][str(Path(__file__).resolve())]=own_hash
        report['thin_ring_initialization'].update(candidate_mask_tolerance_pixels=1,
            final_bundle_renderer_unchanged=True,final_audit_renderer_unchanged=True,
            reference_initializer_equivalence_claimed=False)
        path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
