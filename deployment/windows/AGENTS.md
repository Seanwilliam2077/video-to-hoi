# Windows runtime maintenance

- This directory holds configuration, not a second environment. The only active
  runtime is the repository root `.venv`; never create `deployment/windows/.venv`.
- Prefer `tools/windows/setup.ps1`. Before any direct `uv sync` maintenance,
  dot-source `tools/windows/activate.ps1` from the repository root. It sets the
  absolute `UV_PROJECT_ENVIRONMENT` to the root environment. Check that value.
- Keep default setup offline and reuse the preserved wheels, model receipts,
  pinned sources and package cache. `object-postprocess` is optional; do not
  install its simplification/texture packages for the default NVIDIA entry.
- Preserve official generation, segmentation and complete conversion methods.
  Resource batching must retain all frames and the original objective, schedule,
  precision and sampling settings. Record its validation and limitations.
- The owner permits the Kaolin penetration fallback; equivalence is unproven.
  Historical convex/JPEG/custom-fit results are experiments, not official baselines.
- Owner preference (2026-10-11): iteration and comparisons stop at native
  inference, preview videos and prediction diagnostics. Run expensive official
  human conversion/submission export only after explicit submission confirmation.
  Do not append automatic export or packed-export rendering to a comparison.
