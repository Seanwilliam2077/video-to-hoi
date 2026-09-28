# CPython 3.11 companion

This archive supplies interpreter-specific Linux x86_64 wheels for the same
selected package versions as the CPython 3.10 runtime collection. Universal,
compatible `abi3`, and source distributions are reused by exact filename and hash.
PyAV is the explicit exception: the upstream package selects `av==18.0.0` for
Python 3.11 and `av==17.0.0` for Python 3.10.

No dependency was installed, built, imported for inference or tested on a GPU.
The PyTorch v2.5.1 upstream Dockerfile defaults to Python 3.11, but the interpreter
inside the recorded image and the host driver/GPU have not been runtime-verified.

## Assemble the profile without mixing CPython 3.10 wheels

1. Extract the published base archive as `runtime-cp310-cu124/`.
2. Extract this companion as `runtime-cp311-cu124/`.
3. From the companion directory, run:

   ```sh
   python assemble_runtime_cp311.py --base ../runtime-cp310-cu124 \
     --companion . --output ../runtime-cp311-ready --allow-missing
   ```

   This copies only the shared files explicitly listed in the companion manifest
   and the new CPython 3.11 files. It does not copy CPython 3.10-specific wheels.
   It preserves the required source/license companions and writes an exact list
   of missing files, including distributions withheld from public redistribution.
4. Obtain missing files directly from their pinned official origins, subject to
   their upstream terms. For a reused file, use the base acquisition manifest and
   the base helper's `--names PACKAGE` option. For a new interpreter-specific file,
   use the companion manifest. Put each file under the extraction root stated by
   that manifest, then rerun the assembly command without `--allow-missing`.
5. Build and validate the module's separate environment on the actual server.
   The assembly command is a file integrity check, not an installation or a
   confirmation of runtime compatibility. Image layers, system packages and
   native compilation still need to be prepared and checked remotely.

`base_manifest_sha256` refers to the exact base acquisition manifest embedded in
the immutable published CPython 3.10 archive. That snapshot recorded TensorRT as
still downloading at packaging time; the repository's later acquisition ledger
records its final local completion. The expected TensorRT origin/hash is the same
in both. The companion also records a separate current-ledger digest so these two
different records cannot be confused.

All redistribution exclusions from the base apply here, including SMPL-X,
proprietary NVIDIA components, Triton's bundled NVIDIA development utilities, and
native copyleft binaries without established corresponding-source coverage.
Track 2 data and derivatives remain prohibited in every stage.
