"""Portable launcher checks; subprocess installation/inference boundaries stay synthetic."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


class PortableSetupTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "scripts/runtime.py"
        self.assertTrue(path.is_file(), "portable runtime helper is not implemented")
        sys.path.insert(0, str(path.parent))
        self.addCleanup(lambda: sys.path.remove(str(path.parent)))
        spec = importlib.util.spec_from_file_location("portable_runtime_test", path)
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)
        self.temp = tempfile.TemporaryDirectory(prefix="ochtagon space ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.run = self.directory / "new run"
        self.lock = self.directory / "cpu.lock"
        self.lock.write_text("synthetic==1 --hash=sha256:" + "0" * 64)
        self.facts = {"implementation": "CPython", "version": "3.11.9", "system": "Linux",
                      "machine": "x86_64", "bits": 64, "libc": "glibc", "libc_version": "2.35", "macos": ""}
        self.calls = []

    def fake_probe(self, python, env=None):
        return dict(self.facts, prefix=str(self.run / "environment") if "environment" in str(python) else "base")

    def fake_step(self, command, run_root, label, env):
        self.calls.append((command, env))
        if "venv" in command:
            executable = self.run / "environment/bin/python"
            executable.parent.mkdir(parents=True)
            executable.write_text("synthetic executable")

    def patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(self.runtime, "LOCKS", {"linux": self.lock}))
        stack.enter_context(patch.object(self.runtime, "probe", side_effect=self.fake_probe))
        stack.enter_context(patch.object(self.runtime, "run_step", side_effect=self.fake_step))
        stack.enter_context(patch.object(self.runtime, "versions", return_value={"numpy": "1.26.4"}))
        stack.enter_context(patch.object(self.runtime.verify, "verify_package", return_value={"valid": True}))
        return stack

    def test_rejects_unsupported_platforms_and_versions(self):
        for updates, requested in [({"machine": "aarch64"}, "auto"), ({"libc_version": "2.27"}, "linux"),
                                   ({"version": "3.12.0"}, "auto"), ({}, "windows"),
                                   ({"system": "Darwin", "machine": "arm64", "macos": "11.7"}, "auto")]:
            with self.subTest(updates=updates), self.assertRaises(RuntimeError):
                self.runtime.select_profile(dict(self.facts, **updates), requested)

    def test_setup_needs_no_target_data_and_never_overwrites(self):
        with self.patches():
            self.runtime.ensure_setup(self.run, sys.executable, "linux")
            self.assertTrue((self.run / "setup.json").is_file())
            install, child_env = next(call for call in self.calls if "install" in call[0])
            self.assertIn("--require-hashes", install)
            self.assertIn("--only-binary=:all:", install)
            self.assertEqual(Path(install[install.index("--cache-dir") + 1]), self.run / "pip-cache")
            self.assertEqual(child_env["TMPDIR"], str(self.run / "temp"))
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.runtime.ensure_setup(self.run, sys.executable, "linux")

    def test_install_failure_preserves_run_without_success_receipt(self):
        original_env = dict(os.environ)
        def fail_install(command, *arguments):
            if "install" in command:
                raise RuntimeError("install failed")
            self.fake_step(command, *arguments)
        with self.patches(), patch.object(self.runtime, "run_step", side_effect=fail_install):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                self.runtime.ensure_setup(self.run, sys.executable, "linux")
        self.assertTrue(self.run.is_dir())
        self.assertFalse((self.run / "setup.json").exists())
        self.assertEqual(dict(os.environ), original_env)

    def test_reuse_requires_matching_lock_and_installed_versions(self):
        with self.patches():
            self.runtime.ensure_setup(self.run, sys.executable, "linux")
            self.calls.clear()
            self.runtime.ensure_setup(self.run, sys.executable, "linux", reuse=True)
            self.assertTrue(any("check" in command for command, _ in self.calls))
            self.assertFalse(any("install" in command for command, _ in self.calls))
            with patch.object(self.runtime, "versions", return_value={"numpy": "changed"}):
                with self.assertRaisesRegex(RuntimeError, "versions"):
                    self.runtime.ensure_setup(self.run, sys.executable, "linux", reuse=True)
            self.lock.write_text("changed lock")
            with self.assertRaisesRegex(RuntimeError, "receipt"):
                self.runtime.ensure_setup(self.run, sys.executable, "linux", reuse=True)

    def test_arbitrary_existing_directory_cannot_be_reused(self):
        self.run.mkdir()
        with self.patches(), self.assertRaisesRegex(RuntimeError, "setup receipt"):
            self.runtime.ensure_setup(self.run, sys.executable, "linux", reuse=True)

    def test_child_temp_is_run_local_without_parent_environment_changes(self):
        (self.run / "temp").mkdir(parents=True)
        before = dict(os.environ)
        env = self.runtime.child_environment(self.run)
        env["PYTHONPATH"] = "should be ignored by -I"
        result = subprocess.run([sys.executable, "-I", "-B", "-c",
                                 "import json,tempfile,sys;print(json.dumps([tempfile.gettempdir(),sys.flags.isolated]))"],
                                env=env, text=True, capture_output=True, check=True)
        observed, isolated = json.loads(result.stdout)
        self.assertEqual(Path(observed).resolve(), (self.run / "temp").resolve())
        self.assertEqual(isolated, 1)
        self.assertEqual(dict(os.environ), before)

    def test_canonical_newlines_preserve_labels_and_refuse_existing_output(self):
        expected = b"frame,label\r\n0,H\r\n1,D\r\n"
        digest = hashlib.sha256(expected).hexdigest()
        for index, content in enumerate([expected, expected.replace(b"\r\n", b"\n")]):
            raw, final = self.directory / f"raw{index}.csv", self.directory / f"final{index}.csv"
            raw.write_bytes(content)
            self.runtime.canonicalize_csv(raw, final, digest)
            self.assertEqual(final.read_bytes(), expected)
            self.assertEqual(raw.read_bytes(), content)
            with self.assertRaises(FileExistsError):
                self.runtime.canonicalize_csv(raw, final, digest)
            self.assertEqual(final.read_bytes(), expected)
        raw.write_bytes(b"frame,label\n0,A\n1,D\n")
        with self.assertRaisesRegex(RuntimeError, "reference"):
            self.runtime.canonicalize_csv(raw, self.directory / "changed.csv", digest)


if __name__ == "__main__":
    unittest.main()
