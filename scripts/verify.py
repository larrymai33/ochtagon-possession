"""Verify frozen release bytes; optionally validate a CSV without ground truth."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "repro/package-manifest.json"
MANIFEST_SHA256 = "9e2da0822adad6fa436eebd771640f0648f58e9d78f8203d240b98afe6ef7e02"
PAYLOAD_FILES = 41
sys.dont_write_bytecode = True


class VerificationError(ValueError):
    """The release or submission does not match its declared contract."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"Cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Invalid {description}: expected a JSON object")
    return value


def verify_package(package_root, expected_manifest_sha256=MANIFEST_SHA256):
    """Hash the manifest first, then verify its exact set of 41 payload files."""
    package_root = Path(package_root).resolve()
    manifest_path = package_root / MANIFEST_PATH
    if not manifest_path.is_file():
        raise VerificationError(f"Package manifest missing: {manifest_path}")
    actual_manifest = sha256(manifest_path)
    if actual_manifest != expected_manifest_sha256:
        raise VerificationError("Package manifest SHA-256 mismatch; restore the frozen package")
    manifest = read_json(manifest_path, "package manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != PAYLOAD_FILES:
        raise VerificationError(f"Package manifest must bind exactly {PAYLOAD_FILES} payload files")
    expected_paths = {MANIFEST_PATH}
    for name, record in files.items():
        relative = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or relative.is_absolute()
                or ".." in relative.parts or str(relative) != name or name == MANIFEST_PATH):
            raise VerificationError("Package manifest contains an unsafe payload path")
        path = package_root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(package_root):
            raise VerificationError("Package manifest contains an unsafe payload path")
        if not path.is_file():
            raise VerificationError(f"Package payload missing: {name}")
        if (not isinstance(record, dict) or type(record.get("size")) is not int
                or record["size"] < 0 or not isinstance(record.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
            raise VerificationError(f"Invalid package manifest entry: {name}")
        if path.stat().st_size != record["size"]:
            raise VerificationError(f"Package size mismatch: {name}")
        if sha256(path) != record["sha256"]:
            raise VerificationError(f"Package SHA-256 mismatch: {name}")
        expected_paths.add(name)
    observed = set()
    for path in package_root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise VerificationError("Package must not contain symbolic links or junctions")
        if path.is_file():
            observed.add(path.relative_to(package_root).as_posix())
    unexpected = sorted(observed - expected_paths)
    if unexpected:
        raise VerificationError(f"Package contains unexpected file: {unexpected[0]}")
    return {"valid": True, "files_verified": len(files), "manifest_sha256": actual_manifest}


def verify_submission(csv_path, reference_path, source_root):
    """Check reference bytes and the packaged official CSV validator, never scores."""
    reference = read_json(reference_path, "submission reference")
    expected_frames = reference.get("frames")
    expected_hash = reference.get("submission_sha256")
    if (type(expected_frames) is not int or expected_frames <= 0
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None):
        raise VerificationError("Invalid submission reference: frames and submission_sha256 required")
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise VerificationError(f"Submission CSV missing: {csv_path}")
    digest = sha256(csv_path)
    if digest != expected_hash:
        raise VerificationError("CSV SHA-256 differs from the reference submission")
    # Load this exact packaged file rather than any installed/research evaluation module.
    module_path = Path(source_root) / "evaluation/possession_metrics.py"
    try:
        spec = importlib.util.spec_from_file_location("_ochtagon_official_possession_metrics", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError, ValueError) as exc:
        raise VerificationError("Official CSV validator unavailable; use the pinned runtime") from exc
    try:
        frame = module.read_label_csv(csv_path)
        reason = module.validate_submission(frame, expected_frames)
    except Exception as exc:
        raise VerificationError("CSV failed official format validation") from exc
    if reason is not None:
        # Do not expose label values or table rows through validation errors.
        raise VerificationError("CSV failed official format validation")
    if sha256(csv_path) != digest:
        raise VerificationError("CSV changed during official validation")
    return {"valid": True, "frames": len(frame), "sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="also verify this CSV against verification/reference.json")
    args = parser.parse_args(argv)
    try:
        result = {"package": verify_package(ROOT / "package")}
        if args.csv is not None:
            result["submission"] = verify_submission(
                args.csv, ROOT / "verification/reference.json", ROOT / "package/source")
    except (VerificationError, OSError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
