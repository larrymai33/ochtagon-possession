"""Reproduce the frozen CSV using an isolated setup and exact reference check."""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime as rt

INPUTS = {"tactical/BAR-ATH_tactical_atd.xml": "327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a",
          "tactical/BAR-ATH_tactical_atd.txt": "f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431"}


def main():
    parser = rt.common_parser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root.expanduser().resolve()
    for name, digest in INPUTS.items():
        if not (data / name).is_file() or rt.verify.sha256(data / name) != digest:
            raise RuntimeError(f"Tracking file missing or SHA-256 mismatch: {name}")
    reference = rt.verify.read_json(rt.ROOT / "verification/reference.json", "submission reference")
    run = rt.checked_run_root(args.run_root)
    for name in ("native-output", "output", "reproduction.json"):
        if os.path.lexists(run / name):
            raise RuntimeError(f"Run already has {name}; outputs are never overwritten")
    context = rt.ensure_setup(run, args.python, args.profile, reuse=True)
    python, env = context["python"], context["env"]
    native, final = run / "native-output/ochtagon.csv", run / "output/ochtagon.csv"
    native.parent.mkdir()
    final.parent.mkdir()
    source = rt.ROOT / "package/source"
    launch = 'import sys,torch,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("solution.final_observation_ensemble",run_name="__main__")'
    rt.run_step([python, "-I", "-B", "-c", launch, source, "infer", "--model",
                 rt.ROOT / "package/models/observation_gap_v1.pkl", "--expected-sha256", rt.MODEL_SHA256,
                 "--data-dir", data, "--out", native], run, "inference", env)
    validator = ('import torch,importlib.util,json,sys;'
                 's=importlib.util.spec_from_file_location("official",sys.argv[1]);'
                 'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
                 'f=m.read_label_csv(sys.argv[2]);'
                 'print(json.dumps(dict(valid=m.validate_submission(f,int(sys.argv[3])) is None,frames=len(f))))')
    format_result = rt.query(python, validator, (source / "evaluation/possession_metrics.py", native, reference["frames"]), env)
    if format_result != {"valid": True, "frames": reference["frames"]}:
        raise RuntimeError("Native CSV failed official format validation; outputs preserved")
    print(f"Native CSV passes official format validation ({format_result['frames']} frames).", flush=True)
    native_manifest = native.with_suffix(".csv.manifest.json")
    digest = rt.canonicalize_csv(native, final, reference["submission_sha256"])
    rt.run_step([python, "-I", "-B", rt.ROOT / "scripts/verify.py", "--csv", final], run, "final-verification", env)
    receipt = {"schema": "ochtagon-portable-inference-v1", "profile": context["profile"],
               "native_output": "native-output/ochtagon.csv", "native_sha256": rt.verify.sha256(native),
               "native_manifest_sha256": rt.verify.sha256(native_manifest), "output": "output/ochtagon.csv",
               "submission_sha256": digest, "serialization": "CRLF", "frames": reference["frames"],
               "official_format_valid": True, "reference_hash_matched": True,
               "model_sha256": rt.MODEL_SHA256, "package_manifest_sha256": rt.verify.MANIFEST_SHA256,
               "input_sha256": INPUTS, "setup_sha256": rt.verify.sha256(run / "setup.json")}
    rt.write_json(final.with_suffix(".csv.manifest.json"), receipt)
    rt.write_json(run / "reproduction.json", dict(receipt, schema="ochtagon-reproduction-v2", success=True))
    print(f"Exact reference CSV verified: {final}\nReceipt: {run / 'reproduction.json'}")


if __name__ == "__main__":
    raise SystemExit(rt.cli(main))
