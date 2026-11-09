"""Synthetic release checks; never load tracking, ground truth, or model objects."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


class PackageVerificationTests(unittest.TestCase):
    def setUp(self):
        script = ROOT / "scripts" / "verify.py"
        self.assertTrue(script.is_file(), "release verifier has not been implemented")
        spec = importlib.util.spec_from_file_location("release_verify", script)
        self.verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.verify)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name) / "package"
        self.package.mkdir()
        self.files = {}
        for index in range(41):
            name = f"payload/{index}.txt"
            path = self.package / name
            path.parent.mkdir(exist_ok=True)
            data = f"frozen payload {index}\n".encode()
            path.write_bytes(data)
            self.files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest = self.package / "repro" / "package-manifest.json"
        self.manifest.parent.mkdir()
        self.write_manifest()

    def write_manifest(self):
        data = json.dumps({"files": self.files}, sort_keys=True).encode()
        self.manifest.write_bytes(data)
        self.expected_hash = hashlib.sha256(data).hexdigest()

    def check_package(self):
        return self.verify.verify_package(self.package, self.expected_hash)

    def test_accepts_complete_bound_package(self):
        result = self.check_package()
        self.assertEqual(result["files_verified"], 41)
        self.assertEqual(result["manifest_sha256"], self.expected_hash)

    def test_rejects_manifest_tampering(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(self.verify.VerificationError, "manifest.*SHA-256"):
            self.check_package()

    def test_rejects_same_size_payload_tampering(self):
        path = self.package / "payload/0.txt"
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaisesRegex(self.verify.VerificationError, "SHA-256.*payload/0.txt"):
            self.check_package()

    def test_rejects_missing_payload(self):
        (self.package / "payload/0.txt").unlink()
        with self.assertRaisesRegex(self.verify.VerificationError, "missing.*payload/0.txt"):
            self.check_package()

    def test_rejects_unbound_payload(self):
        (self.package / "unbound.py").write_text("print('unbound')\n")
        with self.assertRaisesRegex(self.verify.VerificationError, "unexpected.*unbound.py"):
            self.check_package()

    def test_rejects_manifest_path_escape(self):
        self.files["../outside.txt"] = self.files.pop("payload/0.txt")
        self.write_manifest()
        with self.assertRaisesRegex(self.verify.VerificationError, "unsafe.*path"):
            self.check_package()

    def test_csv_requires_reference_before_access(self):
        with self.assertRaisesRegex(self.verify.VerificationError, "reference"):
            self.verify.verify_submission(Path(self.temp.name) / "absent.csv",
                                          Path(self.temp.name) / "absent.json",
                                          ROOT / "package/source")

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "CSV validation tests require the pinned runtime's numpy and pandas")
    def test_official_validator_accepts_synthetic_csv_and_rejects_bad_frame_order(self):
        csv = Path(self.temp.name) / "sample.csv"
        reference = Path(self.temp.name) / "reference.json"
        for data, valid in [(b"frame,label\n0,H\n1,D\n2,X\n", True),
                            (b"frame,label\n0,H\n2,D\n1,X\n", False)]:
            csv.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            reference.write_text(json.dumps({"frames": 3, "submission_sha256": digest}))
            if valid:
                result = self.verify.verify_submission(csv, reference, ROOT / "package/source")
                self.assertEqual(result, {"valid": True, "frames": 3, "sha256": digest})
            else:
                with self.assertRaisesRegex(self.verify.VerificationError, "official.*validation"):
                    self.verify.verify_submission(csv, reference, ROOT / "package/source")

    def test_rejects_csv_reference_hash_mismatch(self):
        csv = Path(self.temp.name) / "sample.csv"
        csv.write_bytes(b"frame,label\n0,H\n")
        reference = Path(self.temp.name) / "reference.json"
        reference.write_text(json.dumps({"frames": 1, "submission_sha256": "0" * 64}))
        with self.assertRaisesRegex(self.verify.VerificationError, "CSV.*SHA-256"):
            self.verify.verify_submission(csv, reference, ROOT / "package/source")


@unittest.skipUnless(shutil.which("powershell") or shutil.which("pwsh"), "PowerShell unavailable")
class ReproductionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.script = ROOT / "scripts/reproduce.ps1"
        self.assertTrue(self.script.is_file(), "reproduction wrapper has not been implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def invoke(self, run_root):
        return subprocess.run([
            shutil.which("powershell") or shutil.which("pwsh"), "-NoProfile",
            "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(self.script),
            "-DataRoot", str(self.directory), "-RunRoot", str(run_root),
            "-Python311", sys.executable,
        ], text=True, capture_output=True, timeout=30)

    def test_refuses_existing_run_directory_without_changing_it(self):
        run_root = self.directory / "existing"
        run_root.mkdir()
        marker = run_root / "keep.txt"
        marker.write_text("keep")
        result = self.invoke(run_root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RunRoot already exists", result.stderr)
        self.assertEqual(list(run_root.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "keep")

    def test_refuses_run_directory_inside_repository_without_creating_it(self):
        run_root = ROOT / ("test-run-" + uuid.uuid4().hex)
        result = self.invoke(run_root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RunRoot must be outside", result.stderr)
        self.assertFalse(run_root.exists())

    def test_refuses_wrong_tracking_hash_before_creating_run_directory(self):
        tactical = self.directory / "tactical"
        tactical.mkdir()
        (tactical / "BAR-ATH_tactical_atd.xml").write_text("synthetic XML")
        (tactical / "BAR-ATH_tactical_atd.txt").write_text("synthetic tracking")
        run_root = self.directory / "new-run"
        result = self.invoke(run_root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Tracking SHA-256 mismatch", result.stderr)
        self.assertFalse(run_root.exists())


if __name__ == "__main__":
    unittest.main()
