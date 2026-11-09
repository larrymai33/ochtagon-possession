# ochtagon possession inference

Reproduce the **ochtagon** possession CSV for More Than A Hack 2027 using the frozen `observation_gap_v1` model. It uses automatic tactical tracking and a pretrained tree/two-network ensemble. The repository contains the model, its complete inference source, pinned dependencies, and verification tools.

**Exact CSV reproduction verified on Windows and Ubuntu WSL on November 9, 2026.** Both workflows used a clean checkout and freshly installed pinned environment on the same physical host, passed the official validator, and matched the reference bytes. See [current platform evidence](verification/PLATFORMS.md). macOS wheels are verified; execution on a Mac remains untested.

The competition data must be obtained separately by an authorized participant. The model is already trained; reproduction runs inference only.

## Choose a platform

All profiles run inference on CPU. A CUDA GPU is not required.

| Profile | Requirements | Verification status |
| --- | --- | --- |
| `windows` | Windows AMD64, CPython **3.11.0** | Fresh setup and exact CSV reproduction passed through Command Prompt |
| `linux` | Linux x86_64, glibc **2.28+**, CPython **3.11.x** | Fresh setup and exact CSV reproduction passed on local Ubuntu WSL |
| `macos` | Apple Silicon/arm64, macOS **12+**, CPython **3.11.x** | All 19 wheels downloaded and verified; hardware execution untested |

The tested code commit is `7f27085bfc9024391b22deda827b27ef1fc39f42`. Windows passed all 20 release tests; Linux passed 16, with four PowerShell-specific tests skipped. The Linux run used CPython 3.11.18, glibc 2.43 and Torch 2.11.0+cpu. These results cover the two tested environments on one physical host, not every compatible machine. Details are in [verification/PLATFORMS.md](verification/PLATFORMS.md).

Intel Macs and Linux ARM are not configured. Windows uses the original [nineteen-wheel lock](package/repro/requirements-windows-cp311.lock); Linux and macOS use the platform locks and source records under [requirements/](requirements/README.md). Setup validates the selected interpreter and platform before installation.

Install the required Python and Git first; the scripts create environments but do not install a Python interpreter. Allow at least **15 GB free on the drive containing the run folder** for downloads, installation, temporary files and outputs. Windows wheel downloads alone are approximately 2.84 GB. The run filesystem must support hard links. Clone the repository, which currently requires access because public release is pending:

```text
git clone https://github.com/larrymai33/ochtagon-possession.git
cd ochtagon-possession
```

Run the following commands from the checkout root. Replace example paths with absolute paths on your machine; keep the run folder outside the checkout.

## Set up without match data

The setup script creates a new environment, installs the selected hash-pinned dependencies, checks them and writes `<RunRoot>/setup.json`. It needs no competition files. Installation temporary files and the pip cache stay within the run folder. Choose a new, absent run folder; an existing or failed installation is not overwritten.

Windows — the same command works in Command Prompt, PowerShell and Git Bash when `python` selects CPython 3.11.0:

```text
python scripts/setup.py --run-root "E:/ochtagon-run" --profile windows
```

Linux or macOS — `auto`, the default, selects the matching profile:

```sh
python3.11 scripts/setup.py --run-root "/absolute/path/ochtagon-run"
```

To select another installed interpreter explicitly, add `--python "C:/path/to/Python311/python.exe"` on Windows or `--python "/absolute/path/to/python3.11"` on Linux/macOS. Windows requires exactly 3.11.0: `python` or `py -3.11` can select a later patch release, so use `--python` when necessary. You can launch the script with another compatible modern Python and select the required target interpreter this way. You can also specify `--profile linux` or `--profile macos`. No environment activation is needed.

## Generate the CSV

Arrange the separately obtained, authorized input files as follows:

```text
<DataRoot>/tactical/BAR-ATH_tactical_atd.xml
<DataRoot>/tactical/BAR-ATH_tactical_atd.txt
```

Windows — Command Prompt, PowerShell or Git Bash:

```text
python scripts/reproduce.py --run-root "E:/ochtagon-run" --data-root "E:/authorized-data" --profile windows
```

Linux or macOS:

```sh
python3.11 scripts/reproduce.py --run-root "/absolute/path/ochtagon-run" --data-root "/absolute/path/authorized-data"
```

This command can create the environment itself when the run folder is absent, so the setup-only step is optional. After running `setup.py`, use `reproduce.py` with the same run folder. If setup used `--python`, pass the same `--python` path to reproduction. It checks the completed setup's receipt, profile, interpreter, dependency versions, lock and package hashes before reuse. It refuses unrelated or failed run folders and never overwrites outputs. The same `--python` and `--profile` options are available on both scripts.

The existing Windows PowerShell route also remains available as an all-in-one command. It cannot reuse a folder created by `setup.py`; choose a different, new run folder for this route:

```powershell
.\scripts\reproduce.ps1 -DataRoot 'E:\authorized-data' -RunRoot 'E:\new-ochtagon-run' -Python311 'C:\Python311\python.exe'
```

## Outputs and verification

The Python workflow runs the frozen model twice and requires identical labels. Its final submission file is **`<RunRoot>/output/ochtagon.csv`**, with a separate `ochtagon.csv.manifest.json` receipt. It preserves the frozen inference output and its original receipt under `<RunRoot>/native-output/`. The outer script only serializes newlines as CRLF for the final file, so operating-system newline differences do not change the submission bytes. Frame numbers and labels are unchanged.

The final CSV must match the reference SHA-256 in [verification/reference.json](verification/reference.json) and pass the official format validator: header `frame,label`, 199,319 ordered rows, frames 0 through 199318 and labels `H`, `A`, `D` or `X`. Any reference mismatch fails reproduction, even on a platform with an available setup profile. Successful Python runs write `<RunRoot>/reproduction.json`. See [the reproduction record](verification/REPRODUCTION.md) for the original tested checkout and results.

Package-only verification uses the Python standard library: run `python scripts/verify.py` on Windows or `python3.11 scripts/verify.py` on Linux/macOS. CSV validation needs the installed environment's NumPy and pandas. For example, in PowerShell:

```powershell
& 'E:/ochtagon-run/environment/Scripts/python.exe' scripts/verify.py --csv 'E:/ochtagon-run/output/ochtagon.csv'
```

On Linux or macOS:

```sh
"/absolute/path/ochtagon-run/environment/bin/python" scripts/verify.py --csv "/absolute/path/ochtagon-run/output/ochtagon.csv"
```

These workflows perform integrity, format and reproduction checks. They do not score predictions against target ground truth or compare them with video.

## Repository contents

| Path | Purpose |
| --- | --- |
| `ochtagon.md` | Submission technical document: approach, labels, assets, limitations and reproduction |
| `PROVENANCE.md` | Frozen model/source identities and training-data provenance |
| `scripts/` | Environment setup, reproduction and verification |
| `requirements/` | Linux/macOS CPU locks and dependency source records |
| `verification/` | Reference CSV identity and fresh-checkout reproduction record |
| `package/models/` | Frozen model and manifest |
| `package/source/` | Complete executable inference source |
| `package/repro/` | Exact dependency lock, package manifest and source archive |
| `package/evidence/` | Aggregate external experiment results, fit receipt and reviews |
| `package/THIRD_PARTY_NOTICES.md` | SkillCorner attribution and dependency notices |

The `package/` directory preserves the November 3, 2026 candidate byte for byte. Its historical records describe that snapshot; later reproduction evidence belongs to this repository's verification records.

## Evidence and scope

The earlier Windows PowerShell rehearsal at commit `eba959d33ca8f3a1cd52ab4e997455e871e24c26` passed exact reproduction and 13 release tests. Its [original reproduction record](verification/REPRODUCTION.md) remains unchanged alongside the later platform evidence.

On sixteen repeatedly used external development matches, mean spellF1 was **0.748792**, compared with **0.751613** for the retained ensemble. Under fixed synthetic missing-ball gaps, it was **0.729366**, compared with **0.715707**. This met the predeclared engineering similarity criterion; it is not independent evidence of target performance or advancement.

The packaged candidate was trained and selected using the declared external data. Earlier research included target-ground-truth tuning, provided-video fitting and a documented public-GT disclosure. Those activities are not inputs to this inference workflow, but a new repository does not erase that history or resolve competition eligibility. See [the technical document](ochtagon.md), [provenance](PROVENANCE.md), and the [superseding organizer clarification](package/RULES_CLARIFICATION_2026-11-03.md).

This repository is prepared for review. Public release and form submission are separate actions.
