"""Use the original temporal pipeline with per-candidate image admission.

No ellipse proposals or candidate silhouette expansion are used by this entry.
The original FoundationPose/mask-proposal initializer and bundle remain intact.
"""
import json
from pathlib import Path
import sys


def main():
    from v2hoi.native import sha256
    from v2hoi.temporal_ring_supported_runtime import main as supported
    source = Path(__file__).resolve()
    source_hash = sha256(source)
    supported(ring_hypotheses=False)
    if '--worker' in sys.argv:
        if sha256(source) != source_hash:
            raise ValueError('Supported-branch entry changed during inference')
        output = Path(sys.argv[sys.argv.index('--output')+1])
        path = output/'result.json'
        report = json.loads(path.read_text())
        report['inputs'][str(source)] = source_hash
        report['image_supported_initialization'].update(
            ellipse_proposals=False, candidate_mask_tolerance_pixels=0)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
