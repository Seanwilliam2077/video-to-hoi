"""Read-only producer/runtime crop-CLI provenance audit; writes a new receipt only."""
from pathlib import Path
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import subprocess
import sys

import numpy as np


TASK = Path(__file__).resolve().parent
REPO = Path("C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_function(source):
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "digest")
    namespace = {"Path": Path, "hashlib": hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "isolated_sha256_helper", "exec"), namespace)
    return namespace["digest"]


def without_digest(source):
    tree = ast.parse(source)
    tree.body = [item for item in tree.body if not (isinstance(item, ast.FunctionDef) and item.name == "digest")]
    return ast.dump(tree, include_attributes=False)


def main():
    current = TASK / "runtime02/tools/prepare_refined_egohand_crops.py"
    current_text = current.read_text()
    current_hash = sha(current)
    git_blob = subprocess.check_output(["git", "show", "28deb40:tools/prepare_refined_egohand_crops.py"], cwd=REPO)
    if hashlib.sha256(git_blob).hexdigest() != current_hash or sha(REPO / "tools/prepare_refined_egohand_crops.py") != current_hash:
        raise ValueError("Current repository/commit/runtime comparison identities changed")
    inputs = {item["episode"]: item for item in read(TASK / "inputs-manifest01.json")["episodes"]}
    sys.path.insert(0, str(TASK / "runtime02/src"))
    from v2hoi.refined_egohand import build_refined_crops
    import v2hoi.refined_egohand as crop_module
    rows = []
    for episode in (16, 11):
        item = inputs[episode]
        root = TASK / f"crops/episode_{episode:06d}"
        receipt_path = root / "crop-policy-receipt.json"
        receipt = read(receipt_path)
        producer = root / "source_snapshot/prepare_refined_egohand_crops.py"
        producer_text = producer.read_text()
        if sha(producer) != receipt["source_sha256"]["tools/prepare_refined_egohand_crops.py"]:
            raise ValueError("Archived actual crop producer no longer matches its own receipt")
        same = without_digest(producer_text) == without_digest(current_text)
        if not same:
            raise ValueError("Crop CLI changed beyond SHA256 helper implementation")
        old_digest, new_digest = digest_function(producer_text), digest_function(current_text)
        checks = []
        files = [(Path(item["original_observations"]), receipt["input_sha256"]["observations"]),
                 (Path(item["manifest"]), receipt["input_sha256"]["manifest"]),
                 (Path(item["video"]), receipt["source_video_sha256"]),
                 (root / "crop_boxes.npy", receipt["crop_boxes_sha256"]),
                 (root / "crop_anchors.npz", receipt["crop_anchors_sha256"])]
        modules = []
        for relative, expected in receipt["source_sha256"].items():
            archived = root / "source_snapshot" / Path(relative).name
            files.append((archived, expected))
            if relative.startswith("src/"):
                runtime_hashes = {name: sha(TASK / name / relative) for name in ("runtime01", "runtime02")}
                if sha(archived) != expected or set(runtime_hashes.values()) != {expected}:
                    raise ValueError("Imported crop-computation module changed")
                modules.append({"relative_path": relative, "producer_snapshot": str(archived),
                                "sha256": expected, "runtime_hashes": runtime_hashes})
        for path, expected in files:
            first, second = old_digest(path), new_digest(path)
            if first != expected or second != expected:
                raise ValueError("Producer/later hashing helper changed an input or artifact identity")
            checks.append({"path": str(path), "producer_and_later_digest_equal": True, "sha256": expected})
        with np.load(item["original_observations"], allow_pickle=False) as archive:
            observations = {key: archive[key] for key in archive.files}
        boxes, arrays, computed_report = build_refined_crops(observations)
        if not np.array_equal(boxes, np.load(root / "crop_boxes.npy", allow_pickle=False)):
            raise ValueError("In-memory unchanged crop computation did not reproduce actual crop boxes")
        array_checks = []
        with np.load(root / "crop_anchors.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("Crop trace array keys changed")
            for key, value in arrays.items():
                actual = archive[key]
                equal = np.array_equal(actual, value, equal_nan=True) if actual.dtype.kind in "fc" else np.array_equal(actual, value)
                if actual.dtype != value.dtype or actual.shape != value.shape or not equal:
                    raise ValueError("In-memory crop trace differs from archived input")
                array_checks.append({"key": key, "dtype": str(actual.dtype), "shape": list(actual.shape),
                                     "values_equal_including_nan": True, "payload_bytes_equal": actual.tobytes() == value.tobytes()})
        rows.append({"episode": episode, "producer_recorded_at": receipt["recorded_at"],
                     "crop_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
                     "actual_producer": {"path": str(producer), "sha256": sha(producer), "bytes": producer.stat().st_size},
                     "only_digest_function_differs_from_later_runtime": same,
                     "imported_numerical_modules": modules, "digest_equivalence_on_actual_inputs_and_artifacts": checks,
                     "in_memory_crop_boxes_exactly_reproduced": True, "crop_trace_arrays": array_checks,
                     "core_report_fields_reproduced": all(receipt.get(key) == value for key, value in computed_report.items())})
    diff = ''.join(difflib.unified_diff((TASK / 'crops/episode_000016/source_snapshot/prepare_refined_egohand_crops.py').read_text().splitlines(True),
                                      current_text.splitlines(True), fromfile="actual_crop_producer_5e1c", tofile="later_runtime_cli_9078"))
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(), "track": 1, "status": "actual_crop_producer_preserved_and_numeric_equivalence_verified",
        "auditor_sha256": sha(__file__), "episodes": rows,
        "later_cli": {"runtime01_sha256": sha(TASK / "runtime01/tools/prepare_refined_egohand_crops.py"),
                      "runtime02_sha256": current_hash, "current_repo_sha256": sha(REPO / "tools/prepare_refined_egohand_crops.py"),
                      "git_28deb40_blob_sha256": hashlib.sha256(git_blob).hexdigest()},
        "exact_cli_diff": diff,
        "source_hash_path_logic": "The CLI hashes Path(__file__) plus ROOT/src modules, where ROOT is the executed CLI's parents[1]. It does not hardcode a separate live REPO path. Each artifact snapshots those bytes after recording their hashes.",
        "numerical_impact": "The only CLI difference is equivalent SHA256 file-reading implementation. All crop calculations/imported modules are identical; actual crop boxes and trace arrays are reproduced in memory without writing replacements.",
        "staging_implication": "Use the receipt-bound actual 5e1c producer snapshot for initial tight crops. The later 9078 runtime/commit copy is not the historical producer and must not replace its receipt hash.",
        "preservation": "No crop, receipt, runtime, inference, native result, evaluation, video or canonical artifact was changed. Only this new audit recipe/receipt is added.",
        "original_launch_argv_not_recovered_by_this_audit": True,
        "official_scores_computed": False, "quality_accepted": False}
    output = TASK / "crop-cli-producer-provenance-audit02.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "episodes": [16, 11], "numeric_crop_inputs_unchanged": True}))


if __name__ == "__main__":
    main()
