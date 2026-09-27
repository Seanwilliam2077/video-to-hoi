# Reviewed model releases

The owner requested a GitHub handoff for CARI4D, SAM3D Body, and SAM3D Objects
after their upstream access approvals. Model acquisition and release packaging
do not install a runtime, deploy a server, or run inference.

## License and provenance

The original Hugging Face asset entries remain unchanged so existing receipts
still match their specification hashes. [model-redistribution.json](model-redistribution.json)
records a separate review tied to the exact asset revision and specification.
The [license files](licenses/) retain their original bytes in Git.

- CARI4D: NVIDIA Open Model Agreement. The current official model card identifies
  the same Aug-25, step-200000 checkpoint and its upstream SHA-256 matches the
  pinned download. The archive includes the full agreement, card, and notice.
- SAM3D Body and Objects: SAM License. Each archive includes the full license
  and original README. The Body archive also preserves the MHR Apache-2.0 license.
- These models retain their own licenses; the project code license does not
  replace them. The release is a team-prepared redistribution, not an upstream
  release or endorsement. Read the included full terms before using the models.

`tools/prepare_bundle.py` continues to exclude gated models by default. Only
`tools/prepare_model_release.py` accepts these three explicitly reviewed model
entries. It verifies all required files, receipt hashes, source identity, and
license bytes before creating the release archives. No cache or credential
directory is included. [Asset preparation status](PREPARATION_STATUS.md) records
what has actually completed; a review entry alone does not mean files are present.

## Download and restore

The [published release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927) contains 11 model parts and four companion files. Its tag is `hf-models-20260927`. Download its `model-release-manifest.json`,
`restore_model_release.py`, and `MODEL_NOTICE.md` assets. Read the notice, then run:

```bash
python restore_model_release.py model-release-manifest.json \
  --parts-dir downloads --output-dir archives --download
```

Python 3.10+ and its standard library are sufficient. The command downloads
missing parts from this repository's release, checks every part's size and
SHA-256, and reconstructs `cari4d.zip`, `sam3d-body.zip`, and `sam3d-objects.zip`.
Completed local parts are reused after verification. An interrupted individual
part is downloaded again. Omit `--download` to work entirely from local parts.
No archive is extracted and no model is loaded by this command.

After verification, extract those three ZIPs into the same project directory
used for the original public-assets handoff and FoundationPose add-on. They carry
the exact `weights/` directory layout and separate `.receipts/<model-id>.json`
files. Per-archive provenance also remains inside each model ZIP. The companion
source repository contains the combined redistribution review.

From that combined project directory, verify the model files without network:

```bash
python tools/fetch_assets.py verify --root . --ids cari4d sam3d-body sam3d-objects
```

The three new models occupy about 18 GB. Keeping downloaded parts, reconstructed
ZIPs, and extracted models simultaneously requires about 54 GB, plus the existing
handoff and filesystem overhead. Keep enough free space for all three copies.

## Produce a release from approved local downloads

```bash
python tools/prepare_model_release.py \
  --asset-root artifacts/preparation --output-dir dist/hf-models-20260927
```

The output includes one ZIP per model, parts no larger than 1,900 MiB,
`model-release-manifest.json`, and `SHA256SUMS`. ZIP storage is uncompressed to
avoid expensive recompression of dense weight tensors. Each part is below the
GitHub release asset size limit. Upload the parts and companions; do not upload
the unsplit large archives or put model blobs into normal Git history.

Each archive includes all required model files, a sanitized receipt, the exact
reviewed license/notice files, and per-file provenance. Revision or license
changes require a new explicit review. A published archive must keep its version
and hashes; use a new release for different bytes.
