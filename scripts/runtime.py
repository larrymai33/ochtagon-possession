"""Standard-library setup, isolation and receipt helpers for portable launchers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("ochtagon_verify", ROOT / "scripts/verify.py")
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)
MODEL_SHA256 = "76c4508ff2091e008de23ddec32ecfc86f989e172916702cadd37511dc68b672"
LOCKS = {"windows": ROOT / "package/repro/requirements-windows-cp311.lock",
         "linux": ROOT / "requirements/linux-cpu.lock", "macos": ROOT / "requirements/macos-arm64.lock"}
RUNTIME_FIELDS = ("implementation", "version", "system", "machine", "bits", "libc", "libc_version", "macos")


def child_environment(run_root=None):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if run_root is not None:
        for name in ("TEMP", "TMP", "TMPDIR"):
            env[name] = str(Path(run_root) / "temp")
    return env


def query(python, code, arguments=(), env=None):
    result = subprocess.run([str(python), "-I", "-B", "-c", code, *map(str, arguments)],
                            env=env or child_environment(), capture_output=True, text=True, shell=False)
    if result.returncode:
        raise RuntimeError(f"Python validation/probe failed (exit {result.returncode}); check the selected runtime")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("Python probe returned invalid metadata") from exc


def probe(python, env=None):
    return query(python, 'import json,platform,struct,sys; print(json.dumps(dict('
                 'implementation=platform.python_implementation(),version=platform.python_version(),'
                 'system=platform.system(),machine=platform.machine(),bits=struct.calcsize("P")*8,'
                 'libc=platform.libc_ver()[0],libc_version=platform.libc_ver()[1],'
                 'macos=platform.mac_ver()[0],prefix=sys.prefix)))', env=env)


def select_profile(facts, requested="auto"):
    profile = {("Windows", "amd64"): "windows", ("Linux", "x86_64"): "linux",
               ("Darwin", "arm64"): "macos"}.get((facts["system"], facts["machine"].lower()))
    if profile is None or (requested != "auto" and requested != profile):
        raise RuntimeError("Unsupported OS/architecture or profile: use Windows AMD64, Linux x86_64, or macOS arm64")
    if facts["implementation"] != "CPython" or facts["bits"] != 64 or not facts["version"].startswith("3.11."):
        raise RuntimeError("A supported 64-bit CPython 3.11 interpreter is required; pass --python")
    if profile == "windows" and facts["version"] != "3.11.0":
        raise RuntimeError("The Windows profile requires exactly CPython 3.11.0")
    numeric = lambda value: tuple(int(part) for part in value.split(".") if part.isdigit())
    if profile == "linux" and (facts["libc"] != "glibc" or numeric(facts["libc_version"]) < (2, 28)):
        raise RuntimeError("The Linux profile requires glibc >= 2.28; musl/Alpine is unsupported")
    if profile == "macos" and numeric(facts["macos"]) < (12, 0):
        raise RuntimeError("The macOS arm64 profile requires macOS >= 12.0")
    return profile


def selected_python(value=None):
    value = value or (sys.executable if sys.version_info[:2] == (3, 11) else shutil.which("python3.11"))
    if not value:
        raise RuntimeError("Install CPython 3.11 and supply its executable with --python")
    path = Path(shutil.which(str(value)) or value).resolve()
    if not path.is_file():
        raise RuntimeError(f"Python executable not found: {path}")
    return path


def checked_run_root(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise RuntimeError("Run root must be outside the repository")
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_step(command, run_root, label, env):
    log = Path(run_root) / "logs" / f"{time.time_ns()}-{label}.log"
    print(f"Running {label}; log: {log}", flush=True)
    with log.open("xb") as output:
        result = subprocess.run(list(map(str, command)), env=env, stdout=output,
                                stderr=subprocess.STDOUT, shell=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); preserved run and log: {log}")


def versions(python, env):
    return query(python, 'import importlib.metadata,json; print(json.dumps('
                 '{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))', env=env)


def ensure_setup(run_root, python=None, profile="auto", reuse=False):
    run_root = checked_run_root(run_root)
    exists = run_root.exists()
    if exists and not reuse:
        raise RuntimeError("Run root already exists; choose a new absent directory")
    verify.verify_package(ROOT / "package")
    base_python = selected_python(python)
    facts = probe(base_python)
    profile = select_profile(facts, profile)
    lock = LOCKS[profile]
    if not lock.is_file():
        raise RuntimeError(f"Profile lock is missing: {lock}")
    identity = {"profile": profile, "runtime": {key: facts[key] for key in RUNTIME_FIELDS},
                "lock_sha256": verify.sha256(lock), "package_manifest_sha256": verify.MANIFEST_SHA256,
                "model_sha256": MODEL_SHA256,
                "launcher_sha256": {name: verify.sha256(ROOT / "scripts" / name)
                                    for name in ("runtime.py", "setup.py", "reproduce.py", "verify.py")}}
    env = child_environment(run_root)
    environment = run_root / "environment"
    executable = environment / ("Scripts/python.exe" if profile == "windows" else "bin/python")
    receipt_path = run_root / "setup.json"
    if exists:
        if not receipt_path.is_file():
            raise RuntimeError("Existing run has no completed setup receipt; use a new run root")
        receipt = verify.read_json(receipt_path, "setup receipt")
        if receipt.get("schema") != "ochtagon-setup-v1" or receipt.get("success") is not True or receipt.get("identity") != identity:
            raise RuntimeError("Setup receipt does not match profile/runtime/lock/package/launcher; use a new run root")
        if any(not (run_root / name).is_dir() for name in ("temp", "pip-cache", "logs")):
            raise RuntimeError("Completed setup directories are missing; use a new run root")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        for name in ("temp", "pip-cache", "logs"):
            (run_root / name).mkdir()
        run_step([base_python, "-I", "-B", "-m", "venv", environment], run_root, "venv", env)
        run_step([executable, "-I", "-B", "-m", "pip", "--isolated", "--cache-dir", run_root / "pip-cache",
                  "install", "--require-hashes", "--only-binary=:all:", "-r", lock], run_root, "install", env)
    if not executable.is_file():
        raise RuntimeError("Setup environment Python is missing; use a new run root")
    installed_runtime = probe(executable, env)
    if ({key: installed_runtime[key] for key in RUNTIME_FIELDS} != identity["runtime"]
            or Path(installed_runtime["prefix"]).resolve() != environment.resolve()):
        raise RuntimeError("Environment interpreter does not match the selected runtime or run root")
    installed = versions(executable, env)
    if exists and installed != receipt.get("installed_versions"):
        raise RuntimeError("Installed environment versions differ from the completed setup receipt")
    run_step([executable, "-I", "-B", "-m", "pip", "--isolated", "check"], run_root, "pip-check", env)
    if not exists:
        verify.verify_package(ROOT / "package")
        receipt = {"schema": "ochtagon-setup-v1", "success": True, "identity": identity,
                   "installed_versions": installed, "base_python": str(base_python), "completed_unix": time.time()}
        write_json(receipt_path, receipt)
    return {"root": run_root, "python": executable, "env": env, "profile": profile, "receipt": receipt}


def canonicalize_csv(source, destination, expected_hash):
    data = Path(source).read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in data:
        raise RuntimeError("Native CSV contains bare carriage returns; refusing serialization")
    with Path(destination).open("xb") as output:
        output.write(data.replace(b"\n", b"\r\n"))
    digest = verify.sha256(destination)
    if digest != expected_hash:
        raise RuntimeError("Canonical CSV differs from the reference; NOT verified for official submission. Outputs preserved.")
    return digest


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", help="CPython 3.11 executable; no activation required")
    parser.add_argument("--profile", choices=("auto", "windows", "linux", "macos"), default="auto")
    return parser


def cli(action):
    try:
        action()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1
    return 0
