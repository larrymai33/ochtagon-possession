"""Create an isolated pinned environment; no match data is needed."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime


def main():
    args = runtime.common_parser(__doc__).parse_args()
    result = runtime.ensure_setup(args.run_root, args.python, args.profile)
    print(f"Setup verified ({result['profile']}). Receipt: {result['root'] / 'setup.json'}")


if __name__ == "__main__":
    raise SystemExit(runtime.cli(main))
