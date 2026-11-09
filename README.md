# ochtagon possession inference

Reproduce the **ochtagon** possession CSV for More Than A Hack 2027 using the frozen `observation_gap_v1` model. It uses automatic tactical tracking and a pretrained tree/two-network ensemble. The repository contains the model, its complete inference source, pinned dependencies, and verification tools.

The competition data must be obtained separately by an authorized participant. The model is already trained; reproduction runs inference only.

## Reproduce

Use Windows AMD64 with **CPython 3.11.0**, PowerShell, Git, internet access for dependency downloads, and a filesystem supporting hard links. Allow at least **15 GB free on the drive containing `RunRoot`** for downloads, installation, temporary files and outputs. The wheel downloads alone are approximately 2.84 GB. Inference runs on CPU; a CUDA GPU is not required.

Clone the repository, which currently requires access because public release is pending:

```powershell
git clone https://github.com/larrymai33/ochtagon-possession.git
Set-Location ochtagon-possession
```

Arrange the authorized input files as follows:

```text
<DataRoot>/tactical/BAR-ATH_tactical_atd.xml
<DataRoot>/tactical/BAR-ATH_tactical_atd.txt
```

Open a fresh PowerShell window in this repository's root. Replace the three paths below. `RunRoot` must be a new, absent folder outside the checkout.

```powershell
.\scripts\reproduce.ps1 `
  -DataRoot 'E:\path\to\authorized-data' `
  -RunRoot 'E:\path\to\new-ochtagon-run' `
  -Python311 'C:\path\to\Python311\python.exe'
```

The script verifies the frozen package, creates an isolated environment, installs the exact nineteen-wheel hash lock, and runs the model twice. Installation temporary files and the pip cache stay within `RunRoot`. Both runs must produce identical labels. It writes:

```text
<RunRoot>/output/ochtagon.csv
<RunRoot>/output/ochtagon.csv.manifest.json
```

The CSV has the header `frame,label` and 199,319 ordered rows, covering frames 0 through 199318 with labels `H`, `A`, `D`, or `X`. The accompanying receipt records input, source, model, and output hashes. Verification checks the CSV structure and its SHA-256 against [verification/reference.json](verification/reference.json). Existing output paths are never overwritten; choose a fresh run folder for each reproduction. See [the reproduction record](verification/REPRODUCTION.md) for the tested checkout and results.

Package-only verification uses the Python standard library:

```powershell
& 'C:\path\to\Python311\python.exe' .\scripts\verify.py
```

CSV validation also runs the official validator, which needs NumPy and pandas. Use the environment installed by the reproduction script:

```powershell
& 'E:\path\to\new-ochtagon-run\environment\Scripts\python.exe' .\scripts\verify.py --csv 'E:\path\to\new-ochtagon-run\output\ochtagon.csv'
```

These commands perform integrity and format checks. They do not score predictions against target ground truth or compare them with video.

## Repository contents

| Path | Purpose |
| --- | --- |
| `ochtagon.md` | Submission technical document: approach, labels, assets, limitations and reproduction |
| `PROVENANCE.md` | Frozen model/source identities and training-data provenance |
| `scripts/` | Environment setup, reproduction and verification |
| `verification/` | Reference CSV identity and fresh-checkout reproduction record |
| `package/models/` | Frozen model and manifest |
| `package/source/` | Complete executable inference source |
| `package/repro/` | Exact dependency lock, package manifest and source archive |
| `package/evidence/` | Aggregate external experiment results, fit receipt and reviews |
| `package/THIRD_PARTY_NOTICES.md` | SkillCorner attribution and dependency notices |

The `package/` directory preserves the November 3, 2026 candidate byte for byte. Its historical records describe that snapshot; later reproduction evidence belongs to this repository's verification records.

## Evidence and scope

On sixteen repeatedly used external development matches, mean spellF1 was **0.748792**, compared with **0.751613** for the retained ensemble. Under fixed synthetic missing-ball gaps, it was **0.729366**, compared with **0.715707**. This met the predeclared engineering similarity criterion; it is not independent evidence of target performance or advancement.

The packaged candidate was trained and selected using the declared external data. Earlier research included target-ground-truth tuning, provided-video fitting and a documented public-GT disclosure. Those activities are not inputs to this inference workflow, but a new repository does not erase that history or resolve competition eligibility. See [the technical document](ochtagon.md), [provenance](PROVENANCE.md), and the [superseding organizer clarification](package/RULES_CLARIFICATION_2026-11-03.md).

This repository is prepared for review. Public release and form submission are separate actions.
