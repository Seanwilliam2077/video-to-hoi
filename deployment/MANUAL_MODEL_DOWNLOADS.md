# Manual model acquisition: expanded human and physics candidates

Prepared 2026-09-28. **No command in this file was executed by Codex.** Model weights downloaded in this handoff: **0**. Automatic access to Google Drive and Dropbox is blocked by organization policy. Use these provenance instructions only through a channel approved by the organization; do not use a proxy, mirror or bypass. A successful download does not grant public redistribution rights.

Do not run upstream fetch_demo_data.sh, bulk folder downloads, or evaluation/training/demo scripts: they may retrieve mixed model/data/body archives or unknown videos. Only the exact generic pretrained files below are candidates for owner acquisition. No Track 2 or unknown-provenance dataset is permitted. Do not load a checkpoint just to inspect it.

## Single-file model commands

These examples assume an already available gdown 5.2.0 in an owner-approved environment. They do not install any package. Run from the root of this handoff. The checkpoint filenames are targets relative to each candidate source directory. Existing files should be verified before any re-download.

```powershell
New-Item -ItemType Directory -Force 'sources/wham/checkpoints', 'sources/smoothnet/manual-intake', 'sources/physpt/models/cliff_hr48' | Out-Null
python -m gdown '1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB' -O 'sources/wham/checkpoints/wham_vit_w_3dpw.pth.tar'
python -m gdown '19qkI-a6xuwob9_RFNSPWf1yWErwVVlks' -O 'sources/wham/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
python -m gdown '1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y' -O 'sources/wham/checkpoints/hmr2a.ckpt'
python -m gdown '1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT' -O 'sources/wham/checkpoints/dpvo.pth'
python -m gdown '1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo' -O 'sources/wham/checkpoints/yolov8x.pt'
python -m gdown '1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso' -O 'sources/wham/checkpoints/vitpose-h-multi-coco.pth'
python -m gdown '101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK' -O 'sources/smoothnet/manual-intake/aist-vibe.pth.tar'
python -m gdown '1ZketGlY4qA3kFp044T1-PaykV2llNUjB' -O 'sources/smoothnet/manual-intake/h36m-fcn.pth.tar'
python -m gdown '106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k' -O 'sources/smoothnet/manual-intake/pw3d-spin.pth.tar'
curl.exe --fail --location 'https://www.dropbox.com/scl/fi/1zyefhokc2seull2un8bt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt?rlkey=0hidvwxdi7g7770it04ncalyh&dl=1' --output 'sources/physpt/models/cliff_hr48/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt'
```

The two WHAM core files are alternative checkpoints; upstream demo.yaml selects wham_vit_bedlam_w_3dpw.pth.tar. DPVO is an optional profile dependency. The three SmoothNet download IDs are different pretrained families; their README gives inconsistent example window sizes. The manual-intake paths deliberately do not imply a validated runtime window/configuration. Preserve the original filename and its model card before selecting window 8/16/32/64.

## Folder-only official sources

The official repositories expose no stable single-file ID for these entries in the reviewed source. Open the official folder manually through an approved channel and select each exact file. Do not download entire folders: PhysPT's folder in particular mixes models and data. No invented gdown ID or guessed direct URL is provided.

| Candidate | Official folder | Exact local target relative to candidate root |
|---|---|---|
| gvhmr | [Official source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) | `inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt` |
| gvhmr | [Official source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) | `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` |
| gvhmr | [Official source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) | `inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth` |
| gvhmr | [Official source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) | `inputs/checkpoints/yolo/yolov8x.pt` |
| gvhmr | [Official source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) | `inputs/checkpoints/dpvo/dpvo.pth` |
| htd-refine | [Official source](https://drive.google.com/drive/folders/1Rg6ajhTRwebUZ_coSvfTlxUfPMD0CR9c?usp=sharing) | `inputs/checkpoints/pvanet.pt` |
| htd-refine | [Official source](https://drive.google.com/drive/folders/1Rg6ajhTRwebUZ_coSvfTlxUfPMD0CR9c?usp=sharing) | `inputs/checkpoints/yolo/yolov8x.pt` |
| physpt | [Official source](https://www.dropbox.com/scl/fo/x9yor045ztrcv6rav4pz2/AMDO2krghJF_spnxNc5d9nE?rlkey=epj0vkwoafyp48rz7lp5im1o0&dl=0) | `assets/checkpoint/PhysPT.pt` |
| physpt | [Official source](https://www.dropbox.com/scl/fo/x9yor045ztrcv6rav4pz2/AMDO2krghJF_spnxNc5d9nE?rlkey=epj0vkwoafyp48rz7lp5im1o0&dl=0) | `assets/checkpoint/GlobalTrajPredictor.pt` |
| physpt | [Official source](https://www.dropbox.com/scl/fo/x9yor045ztrcv6rav4pz2/AMDO2krghJF_spnxNc5d9nE?rlkey=epj0vkwoafyp48rz7lp5im1o0&dl=0) | `assets/checkpoint/yolov8x.pt` |

## Individually licensed body models and unresolved helper files

Register personally at [SMPL](https://smpl.is.tue.mpg.de/), [SMPLify](https://smplify.is.tue.mpg.de/), [SMPL-X](https://smpl-x.is.tue.mpg.de/) or [MANO / SMPL+H](https://mano.is.tue.mpg.de/) as appropriate. Accept the terms before obtaining the body model privately. Do not put passwords or tokens in command lines. Do not upload these files to a public GitHub repository or Release without a separate written permission that permits redistribution.

The exact expected body-model destinations and the additional regressors/mappings/statistics are enumerated in expanded-human-physics.json. Required genders depend on the final profile. WHAM's body_models.tar.gz and the PhysPT mixed assets folder are **not approved bulk-download recipes**. Model helper files must be identified individually and their provenance and license checked before acquisition or use. The source-only package intentionally omits them.

## Record an owner-supplied model without loading it

For each manually obtained model, retain the official source URL, original filename, acquisition date, byte length, SHA-256 and full accompanying model license/model card. Hash bytes only; never unpickle/load a checkpoint for this receipt. Example for one explicitly selected file:

```powershell
$modelPath = (Resolve-Path -LiteralPath 'sources/wham/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar').Path
$modelFile = Get-Item -LiteralPath $modelPath
$modelReceipt = [ordered]@{
  id = 'wham-main-bedlam'
  original_filename = $modelFile.Name
  source_url = 'https://drive.google.com/file/d/19qkI-a6xuwob9_RFNSPWf1yWErwVVlks/view'
  acquired_at = (Get-Date).ToUniversalTime().ToString('o')
  bytes = $modelFile.Length
  sha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
  provenance_status = 'owner_acquired_pending_review'
  public_redistribution_allowed = $false
  inference_performed = $false
}
$modelReceipt | ConvertTo-Json | Set-Content -LiteralPath 'wham-main-bedlam.owner-receipt.json' -Encoding utf8
```

A byte hash proves file identity, not that it is a valid model, free of mixed data, usable with the selected adapter, or publicly redistributable. Keep private body-model files and receipts outside any public release upload. Submit generic model provenance and license evidence for review before adding it to the public package. No candidate has been installed, inferred, scored, or declared run-ready by this handoff.
