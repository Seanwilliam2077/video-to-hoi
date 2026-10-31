"""Build optional Human previews from verified local packages and published assets.

No media are decoded, and no models or remote payloads are loaded. Files are hashed; the
publication receipt supplies prior remote-byte verification. The browser's pure
validator is reused through Node so CI and rendering share the strict schema.
This changes neither immutable progress records nor acceptance declarations.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hoi_human_catalog.v1"
RELEASE = re.compile(r"https://github\.com/Seanwilliam2077/video-to-hoi/releases/download/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def board_module():
    spec = importlib.util.spec_from_file_location("human_catalog_board", ROOT / "tools/progress_board.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return board_module().load_json(path)


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_catalog(value, *, node="node"):
    script = "const fs=require('node:fs');const api=require(process.argv[1]);const errors=api.validateHumanCatalog(JSON.parse(fs.readFileSync(0,'utf8')));if(errors.length){process.stderr.write(errors.join('\\n'));process.exit(1)}"
    result = subprocess.run([str(node), "-e", script, str(ROOT / "docs/progress-board/delivery-view.js")],
                            input=json.dumps(value, allow_nan=False), text=True, capture_output=True, check=False)
    require(result.returncode == 0, "Invalid Human preview catalog: " + result.stderr.strip())


def asset_ref(asset, base_url, published):
    name = asset["name"]
    require(isinstance(name, str) and NAME.fullmatch(name), "Unsafe published asset name")
    path = Path(asset["path"])
    require(path.name == name and path.is_file() and not path.is_symlink(), "Missing or mismatched local preview/evidence file")
    require(type(asset["bytes"]) is int and asset["bytes"] > 0 and path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"],
            f"Local published asset bytes changed: {name}")
    actual = published.get(name)
    require(actual is not None and actual.get("verified") is True and actual.get("url") == base_url + "/" + name and
            all(actual.get(key) == asset[key] for key in ("bytes", "sha256")),
            f"Publication receipt does not verify exact asset bytes: {name}")
    return {"url": base_url + "/" + name, "sha256": asset["sha256"], "bytes": asset["bytes"]}


def build_entry(package, record, base_url, published):
    episode, count = package["episode"], package["frames"]
    require(package["source_commit"] == record["source_commit"] and record["episode_id"] == f"episode_{episode:06d}" and record["coverage"]["frames"] == count,
            "Package does not bind the exact progress record source/episode/coverage")
    assets = {}
    for asset in package["assets"]:
        for key, suffix in (("checks", "-checks.json"), ("depth_scale", "-depth-scale.json"),
                            ("selection", "-selection.json"), ("input_bindings", "-input-bindings.json")):
            if asset["name"].endswith(suffix):
                require(key not in assets, "Duplicate package evidence role")
                assets[key] = asset
    require(set(assets) == {"checks", "depth_scale", "selection", "input_bindings"}, "Incomplete package evidence")
    evidence = {role: asset_ref(asset, base_url, published) for role, asset in assets.items()}
    checks, scale, selection, bindings = (read_json(assets[key]["path"]) for key in ("checks", "depth_scale", "selection", "input_bindings"))
    source, model = package["source_video_sha256"], package["mhr_model_sha256"]
    require(checks.get("schema_version") == "human_batch_episode_checks.v1" and checks.get("track") == 1 and checks.get("episode") == episode and
            checks.get("frames") == count and checks.get("source_video_sha256") == source and checks.get("mhr_model_sha256") == model and
            checks.get("structural_checks_passed") is True and checks.get("reconstruction_quality_accepted") is False,
            "Native check does not bind this unaccepted complete Human candidate")
    require(scale.get("source_video_sha256") == source and scale.get("episode") == episode and scale.get("status") == "unverified_monocular_scale" and
            scale.get("metric_scale_verified") is False and scale.get("shared_human_object_scale_verified") is False,
            "Scale evidence changed or claims unsupported metric acceptance")
    selected = checks["selected_candidate"]
    require(selected in {"baseline", "refined"} and selected == package["selected_candidate"] == selection.get("selected_candidate") and
            selection.get("episode") == episode and selection.get("frames") == count and selection.get("quality_accepted") is False and
            selection.get("selected_is_exact_candidate_copy") is True and selection.get("selection_reason") == checks["selection_reason"] and
            selection.get("sha256") == checks["native_results"][selected]["parameters_sha256"],
            "Selected native candidate does not bind its decode check")
    require(bindings.get("episode") == episode and bindings.get("track") == 1 and bindings.get("video_frames") == count and
            bindings.get("video_sha256") == source and bindings.get("mhr_model_sha256") == model and bindings.get("source_commit") == record["source_commit"],
            "Input bindings do not identify this exact Human run")
    previews = []
    for asset in package["previews"]:
        require(asset.get("episode") == episode and asset.get("episode_frames") == count and asset.get("source_video_sha256") == source and asset.get("mhr_model_sha256") == model,
                "Preview provenance differs from the native Human result")
        previews.append({"role": asset["role"], **asset_ref(asset, base_url, published),
                         "content_type": asset["content_type"], "coverage": asset["coverage"]})
    guard_keys = ("heldout_observed_frames", "sufficient", "before_px", "after_px", "nonregression")
    guards = [{"side": side, **{key: checks["hand_heldout_nonregression"][side][key] for key in guard_keys}} for side in ("left", "right")]
    return {"record": record, "episode_id": record["episode_id"], "profile": "native_mhr_mediapipe_review.v1", "frame_count": count,
            "source_video_sha256": source, "mhr_model_sha256": model, "archive": asset_ref(package["archive"], base_url, published),
            "selected_candidate": selected, "selection_reason": checks["selection_reason"], "quality_accepted": False,
            "shared_scene_scale_verified": False, "official_export_accepted": False, "evidence": evidence, "previews": previews, "hand_guards": guards}


def bound_records(root, incoming=None):
    board = board_module()
    episodes = board.load_episodes(root)
    existing = board.read_records(root, episodes)
    by_id = {record["id"]: record for record in existing}
    for record in incoming or []:
        require(record["id"] not in by_id or record == by_id[record["id"]], "Conflicting immutable progress record")
        by_id[record["id"]] = record
    board.validate_records(list(by_id.values()), {episode["id"]: episode for episode in episodes})
    return by_id, {episode["id"]: episode for episode in episodes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate committed catalog and exact registry bindings without media")
    parser.add_argument("--packages", type=Path, help="Local packaging packages.json")
    parser.add_argument("--records", type=Path, help="Exact submitted progress record export envelope")
    parser.add_argument("--release-base-url")
    parser.add_argument("--publication-receipt", type=Path, help="Passing receipt of exact remote asset-byte verification")
    parser.add_argument("--node", default="node", help="Node executable for the shared browser schema validator")
    args = parser.parse_args(argv)
    try:
        output = args.output or args.root / "tracking/human-catalog.json"
        catalog = read_json(output) if output.exists() else {"schema_version": SCHEMA, "entries": []}
        incoming = []
        if args.check:
            require(not any((args.packages, args.records, args.release_base_url, args.publication_receipt)), "--check cannot build or supply alternate records")
        else:
            require(all((args.packages, args.records, args.release_base_url, args.publication_receipt)), "Building requires packages, records, release URL and publication receipt")
            base_url = args.release_base_url.rstrip("/")
            require(RELEASE.fullmatch(base_url), "Only this project's immutable GitHub Release download base is supported")
            records = read_json(args.records)
            require(set(records) == {"schema_version", "records"} and records["schema_version"] == "hoi_progress.v1" and isinstance(records["records"], list), "Invalid record export envelope")
            incoming = records["records"]
        by_id, episodes = bound_records(args.root, incoming)
        validate_catalog(catalog, node=args.node)
        if not args.check:
            publication = read_json(args.publication_receipt)
            require(publication.get("schema_version") == "human_delivery_publication.v1" and publication.get("passed") is True and publication.get("all_manifest_assets_verified") is True and publication.get("release_base_url") == base_url and isinstance(publication.get("assets"), list), "A passing exact-release publication receipt is required")
            published = {item["name"]: item for item in publication["assets"]}
            require(len(published) == len(publication["assets"]), "Duplicate published asset identity")
            package_set = read_json(args.packages)
            require(package_set.get("schema_version") == "human_delivery_set.v1", "Invalid package set")
            index = {entry["record"]["id"]: entry for entry in catalog["entries"]}
            for package in package_set["packages"]:
                require(publication.get("source_commit") == package["source_commit"] and publication.get("delivery_id") == package["delivery_id"], "Publication receipt belongs to a different delivery or code revision")
                matches = [record for record in incoming if record["id"] == f"{package['delivery_id']}-ep{package['episode']:02d}"]
                require(len(matches) == 1, "Every package needs one exact submitted record")
                entry = build_entry(package, matches[0], base_url, published)
                require(entry["record"]["id"] not in index or index[entry["record"]["id"]] == entry, "Cannot change an existing Human preview record; publish a new result ID")
                index[entry["record"]["id"]] = entry
            catalog = {"schema_version": SCHEMA, "entries": sorted(index.values(), key=lambda entry: entry["record"]["id"])}
            validate_catalog(catalog, node=args.node)
        for entry in catalog["entries"]:
            require(by_id.get(entry["record"]["id"]) == entry["record"], "Human preview does not match the exact registered record")
            require(episodes[entry["episode_id"]]["frames"] == entry["frame_count"], "Human preview does not cover every inventory frame")
        if not args.check:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(output)
        print(json.dumps({"schema_version": SCHEMA, "entries": len(catalog["entries"]), "written": not args.check,
                          "scope": "Published preview metadata only; registry acceptance is unchanged"}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Human preview catalog failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
