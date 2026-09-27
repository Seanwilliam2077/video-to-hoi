"""Verify and package the locked CPython 3.10 Linux x86_64 host wheels.

This tool only reads local wheel files. It does not install packages or access
the network. GPU and model-container dependencies are outside its scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CORE_LOCK = REPO / "deployment/requirements/requirements.core.lock"
BOOTSTRAP_LOCK = REPO / "deployment/requirements/requirements.bootstrap.txt"
WHEELHOUSE = REPO / "artifacts/linux-cp310-wheelhouse"
OUTPUT = REPO / "dist/linux-cp310-wheelhouse.zip"
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;#]+)")
MANYLINUX = re.compile(r"manylinux_2_(\d+)_x86_64")
CPYTHON = re.compile(r"cp3(\d+)")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class WheelhouseError(ValueError):
    """Incomplete, incompatible, or unapproved wheelhouse input."""


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lock(path: Path) -> dict[str, str]:
    pins = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise WheelhouseError(f"{path}:{line_number}: expected an exact package==version pin")
        name, version = normalize_name(match[1]), match[2]
        if name in pins and pins[name] != version:
            raise WheelhouseError(f"{path}:{line_number}: conflicting versions for {name}")
        pins[name] = version
    if not pins:
        raise WheelhouseError(f"{path}: empty lock")
    return pins


def wheel_filename(filename: str) -> tuple[str, str, set[str]]:
    if not filename.endswith(".whl"):
        raise WheelhouseError(f"non-wheel file: {filename}")
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6):
        raise WheelhouseError(f"invalid wheel filename: {filename}")
    name, version = normalize_name(parts[0]), parts[1]
    py, abi, platform = parts[-3:]
    tags = {f"{p}-{a}-{s}" for p in py.split(".") for a in abi.split(".") for s in platform.split(".")}
    return name, version, tags


def compatible_tag(tag: str) -> bool:
    parts = tag.split("-")
    if len(parts) != 3:
        return False
    py, abi, platform = parts
    if platform != "any":
        if platform not in {"manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"}:
            match = MANYLINUX.fullmatch(platform)
            if match is None or int(match[1]) > 28:
                return False
    if py in {"py3", "py310"}:
        return abi == "none"
    match = CPYTHON.fullmatch(py)
    if match is None:
        return False
    minor = int(match[1])
    return (minor == 10 and abi in {"cp310", "abi3", "none"}) or (minor < 10 and abi == "abi3")


def verify_wheel(path: Path, name: str, version: str, filename_tags: set[str]) -> None:
    if not any(compatible_tag(tag) for tag in filename_tags):
        raise WheelhouseError(f"{path.name}: no CPython 3.10 Linux x86_64 tag for glibc <= 2.28")
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
            wheel_info = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/WHEEL")]
            if len(metadata) != 1 or len(wheel_info) != 1 or metadata[0].split("/")[0] != wheel_info[0].split("/")[0]:
                raise WheelhouseError(f"{path.name}: missing or ambiguous wheel metadata")
            if archive.getinfo(metadata[0]).file_size > 1024 * 1024 or archive.getinfo(wheel_info[0]).file_size > 1024 * 1024:
                raise WheelhouseError(f"{path.name}: oversized wheel metadata")
            package = BytesParser(policy=default).parsebytes(archive.read(metadata[0]), headersonly=True)
            if normalize_name(package.get("Name", "")) != name or package.get("Version") != version:
                raise WheelhouseError(f"{path.name}: package metadata differs from locked name/version")
            wheel_text = archive.read(wheel_info[0]).decode("utf-8")
            declared_tags = {line[5:].strip() for line in wheel_text.splitlines() if line.startswith("Tag: ")}
            if not any(compatible_tag(tag) for tag in declared_tags & filename_tags):
                raise WheelhouseError(f"{path.name}: WHEEL metadata has no matching compatible tag")
    except (OSError, ValueError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        if isinstance(error, WheelhouseError):
            raise
        raise WheelhouseError(f"{path.name}: invalid wheel archive: {error}") from error


def collect(wheelhouse: Path, expected: dict[str, str], allow_colorama: bool) -> tuple[list[dict], list[dict]]:
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise WheelhouseError(f"wheelhouse is not a directory: {wheelhouse}")
    wheels, ignored = {}, []
    for path in sorted(wheelhouse.iterdir(), key=lambda p: p.name.lower()):
        if path.is_symlink() or not path.is_file():
            raise WheelhouseError(f"unexpected wheelhouse entry: {path.name}")
        name, version, tags = wheel_filename(path.name)
        if name == "colorama" and version == "0.4.6" and allow_colorama and name not in expected:
            ignored.append({"filename": path.name, "sha256": sha256(path), "reason": "Windows resolver extra; excluded from ZIP"})
            continue
        if name not in expected:
            raise WheelhouseError(f"unlocked wheel: {path.name}")
        if version != expected[name]:
            raise WheelhouseError(f"{path.name}: lock requires {name}=={expected[name]}")
        if name in wheels:
            raise WheelhouseError(f"multiple wheels for {name}=={version}")
        verify_wheel(path, name, version, tags)
        wheels[name] = {"name": name, "version": version, "filename": path.name,
                        "bytes": path.stat().st_size, "sha256": sha256(path)}
    missing = sorted(set(expected) - set(wheels))
    if missing:
        raise WheelhouseError("missing locked wheels: " + ", ".join(f"{n}=={expected[n]}" for n in missing))
    return sorted(wheels.values(), key=lambda item: item["filename"]), ignored


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    entry = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    entry.compress_type = zipfile.ZIP_STORED
    entry.external_attr = 0o644 << 16
    archive.writestr(entry, data)


def add_file(archive: zipfile.ZipFile, name: str, path: Path) -> None:
    entry = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    entry.compress_type = zipfile.ZIP_STORED
    entry.external_attr = 0o644 << 16
    with archive.open(entry, "w") as target, path.open("rb") as source:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def prepare(wheelhouse: Path, output: Path, core_lock: Path = CORE_LOCK,
            bootstrap_lock: Path = BOOTSTRAP_LOCK, allow_colorama: bool = False) -> dict:
    if wheelhouse.is_symlink():
        raise WheelhouseError(f"wheelhouse must not be a symlink: {wheelhouse}")
    wheelhouse, output = wheelhouse.resolve(), output.resolve()
    if output.suffix.lower() != ".zip" or output.is_relative_to(wheelhouse):
        raise WheelhouseError("output must be a .zip outside the wheelhouse")
    sidecar = output.with_suffix(".json")
    checksum = output.with_suffix(".sha256")
    if any(path.exists() for path in (output, sidecar, checksum)):
        raise WheelhouseError(f"output already exists: {output}, {sidecar}, or {checksum}")
    core, bootstrap = read_lock(core_lock), read_lock(bootstrap_lock)
    expected = dict(core)
    for name, version in bootstrap.items():
        if name in expected and expected[name] != version:
            raise WheelhouseError(f"lock conflict: {name}=={expected[name]} versus {version}")
        expected[name] = version
    wheels, ignored = collect(wheelhouse, expected, allow_colorama)
    manifest = {
        "schema_version": 1,
        "scope": "host/bootstrap wheels only; no PyTorch, CUDA, model-container dependencies, or model weights",
        "target": {"implementation": "CPython", "python": "3.10", "os": "Linux",
                   "architecture": "x86_64", "maximum_manylinux_glibc": "2.28"},
        "locks": [
            {"filename": core_lock.name, "sha256": sha256(core_lock)},
            {"filename": bootstrap_lock.name, "sha256": sha256(bootstrap_lock)},
        ],
        "wheels": wheels,
        "ignored_extras": ignored,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wheelhouse-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        staged_zip = temporary / output.name
        with zipfile.ZipFile(staged_zip, "w", allowZip64=True) as archive:
            add_file(archive, f"requirements/{core_lock.name}", core_lock)
            add_file(archive, f"requirements/{bootstrap_lock.name}", bootstrap_lock)
            for wheel in wheels:
                add_file(archive, f"wheels/{wheel['filename']}", wheelhouse / wheel["filename"])
            add_bytes(archive, "wheelhouse-manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        digest = sha256(staged_zip)
        result = {**manifest, "archive": {"filename": output.name, "sha256": digest, "bytes": staged_zip.stat().st_size}}
        (temporary / sidecar.name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / checksum.name).write_text(f"{digest}  {output.name}\n", encoding="ascii")
        staged_zip.replace(output)
        (temporary / sidecar.name).replace(sidecar)
        (temporary / checksum.name).replace(checksum)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, default=WHEELHOUSE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--core-lock", type=Path, default=CORE_LOCK)
    parser.add_argument("--bootstrap-lock", type=Path, default=BOOTSTRAP_LOCK)
    parser.add_argument("--allow-colorama", action="store_true",
                        help="ignore only an extra colorama==0.4.6 wheel from Windows resolution")
    args = parser.parse_args()
    try:
        result = prepare(args.wheelhouse, args.output, args.core_lock,
                         args.bootstrap_lock, args.allow_colorama)
    except (OSError, WheelhouseError) as error:
        parser.exit(2, f"wheelhouse: {error}\n")
    print(json.dumps({"archive": result["archive"], "wheel_count": len(result["wheels"]),
                      "ignored_extras": result["ignored_extras"]}, indent=2))


if __name__ == "__main__":
    main()
