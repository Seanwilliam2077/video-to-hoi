# Verified Track 1 raw inputs and Body preparation plan

## Scope and identity

- Delivery: `raw-track1-20261015`, 2026-10-15.
- Module/capability: M1 platform, raw-video preparation and handoff planning.
- Executor: Codex on the current Windows workstation; downstream receiver and human visual reviewer are pending. Roles are not permanently assigned.
- Work type: real Track 1 input verification plus support tooling/documentation. No reconstruction model was executed.
- Source base: `df8213a9ae0a02f7ec0586275647ef7470ba7735`; the commit containing this record pins the added code and configuration.
- Gate source SHA-256: `dc3b85fcc45d5402299df351aa4be61c88b0a0ec1c76e1c83e122b1d64b9699d`.
- Intended consumers: the segmentation/camera workers, then the Body worker after the remaining checks in the [preparation plan](../../../docs/body-input-preparation.md).

## Inputs and reproduction

Only Track 1 at dataset revision `5f68335f3acc802033d1e80728c1633197521de8` was acquired. The [source lock](../../../configs/track1-source.json) has 67 upstream entries; the existing downloader omits the optional 23-byte `.gitignore`, so the delivered dataset has 66 verified files. All 30 original videos, 30 frame-index tables and six metadata/documentation files are present. Source files total 513,185,541 bytes.

The CPU gate fully decoded all **30 videos / 16,563 frames**, preserving original zero-based presentation frame IDs and exact PTS/time base. Every frame-index table matched the decoded timeline. All sources were hashed again before publishing readiness. No crop, resize, re-encoding, masking or camera estimation occurred. Camera strings remain labels, not calibration.

```bash
python -m pip install -e ".[video]"
python -m v2hoi.download --videos --revision 5f68335f3acc802033d1e80728c1633197521de8
python tools/prepare_track1_data.py --dataset-root data/v2d/track_1 --source-inventory configs/track1-source.json --output data/prepared/new-check-attempt
```

Use a fresh output directory. The gate returns nonzero and publishes no ready manifest on failure. [inventory.json](inventory.json) records the exact runtime and per-video identities. Total runtime and peak memory were not measured as a benchmark.

## Outputs and locations

| Output | Location | Identity or verification |
| --- | --- | --- |
| Original source files | Local `data/v2d/track_1/`; teammates can reacquire the exact pinned source | Upstream identities in the source lock; actual per-video hashes in inventory.json |
| Full timelines and producer report | Local `data/prepared/track1-20261015/` | Manifest `156096c31ef4411531a2e723502f312e4f639b0247855340cc60d10f42e77030`; SHA256SUMS covers every generated JSON |
| Portable raw-input archive | Local `dist/track1-raw-inputs-20261015.zip` | 514,457,486 bytes; SHA-256 `eee97dd46601356108a6d85b0893f3209d7d9274627fc213e74838e0473b50fa` |
| Raw gate evidence | [checks.json](checks.json) | 66 source files, 30 decodes and 30 table/timeline matches passed |
| Archive receipt | [package.json](package.json) | All 103 listed member hashes re-read and verified from the ZIP |
| Body preparation procedure | [body-input-preparation.md](../../../docs/body-input-preparation.md) | Defines remaining observations, camera estimates, model-specific gate and receiver checks |

The ZIP includes sources, raw checks, timelines, the source lock, verifier and receiver instructions. It has not been uploaded or verified on another machine. Its large files stay outside normal Git. A teammate can alternatively clone the code and independently acquire the same pinned sources without waiting for archive publication.

The explicit CI Python allowlist passed **565 tests** in 200.85 seconds, including 36 raw-video gate tests with independent synthetic fixtures. [Tool verification](tool-validation.json) is separate from the real-video checks above. The unchanged progress registry also validates; no model-stage completion record was created from raw preparation.

## Separate conclusions and next action

| Conclusion | State |
| --- | --- |
| Raw source download / decode / timeline checks | Passed for all 30 episodes |
| Local archive byte verification | Passed |
| Real target-human masks or boxes | Pending |
| Track 1 camera estimates | Pending |
| Native Body input gate / corrected standalone weight layout | Pending implementation and runtime verification |
| Receiver verification on another host | Pending |
| Body model execution, reconstruction quality, native MHR acceptance | Not run / not accepted |
| Full M1 inputs stage / CARI4D / official export | Not accepted by this raw-input delivery |

Next: choose the worker for preprocessing, generate and review real human observations and camera estimates for complete episodes 16 and 12, resolve the standalone Body weight-layout mismatch, then complete receiver preflight. Keep object preparation parallel to standalone Body bootstrap; formal native CARI4D still waits for its real metric mesh and two-class masks. The [plan](../../../docs/body-input-preparation.md) documents exact boundaries and upstream evidence.
