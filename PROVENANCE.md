# Provenance

This repository packages the frozen `observation_gap_v1` candidate for team **ochtagon**. It preserves the candidate under `package/` and adds reproduction tooling and submission documentation outside that directory.

## Frozen identities

| Item | Identity |
| --- | --- |
| Private research source snapshot | `8017a0a46ff873b71084a5396ece2980ac14a708` |
| Candidate directory in that snapshot | `release/observation_gap_v1` |
| Model | `package/models/observation_gap_v1.pkl` |
| Model SHA-256 | `76c4508ff2091e008de23ddec32ecfc86f989e172916702cadd37511dc68b672` |
| Model sidecar SHA-256 | `0fd1c530a5739cf441e2450d106b1c1fc25bd2fbb9f362d803f85c3c88e380b4` |
| Package manifest SHA-256 | `9e2da0822adad6fa436eebd771640f0648f58e9d78f8203d240b98afe6ef7e02` |
| External comparison contract SHA-256 | `8c00114c32c09d9d8fc24aa145458fafb13883c17735b35b88dc7a28d98185ab` |
| External result SHA-256 | `d750c868d400695a88438f36081f9ff4a1a23c4538762e85534325a0e42c43c4` |
| Reference `ochtagon.csv` SHA-256 | `35d905533b696bee3f29da3e6c9f24ebab76780982e9f2325b6e67784dfea303` |

The package manifest binds 41 payload files. These include executable source, a source ZIP, the model and sidecar, runtime pins, notices and aggregate evidence. `scripts/verify.py` checks the packaged files. Original absolute paths in historical receipts identify the research environment; they are not paths that a reproduction machine must recreate.

## Labelled training source

Source: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), revision `4340d274572876239c154c90bc507a9b3250a656`. The upstream LICENSE at that revision contains the MIT notice reproduced in [package/THIRD_PARTY_NOTICES.md](package/THIRD_PARTY_NOTICES.md). No raw SkillCorner data is bundled.

Permitted training matches:

```text
1874553 1886347 1899585 1925299 1953632 1959846 1986691 1996435
2006229 2006363 2007448 2007721 2011166 2013725 2015213 2016236
```

The cohort has 945,508 valid frames at 10 Hz. Labels come from SkillCorner's automatic provider possession phases, mapped algorithmically to `H`/`A` within phase intervals and `D` in uncovered in-period frames. The team added no independent human annotations to this cohort. We do not claim that these labels are independently adjudicated ground truth. [ochtagon.md](ochtagon.md) describes the mapping, fold boundaries, augmentation and limitations.

The final model was fitted on all sixteen matches after the four-fold external comparison. It did not warm-start from a target-trained model. The retained HGB, temporal-network, blend and decoder settings have documented external-selection lineage; the early HGB report has a less complete provenance record than later hash-bound experiments. The detailed [review](package/evidence/PROVENANCE_REVIEW.md) preserves that limitation. Metrica sample data, SPORTec and quarantined SkillCorner matches are not training dependencies of this model.

## Inference inputs and runtime

Inference requires the separately obtained competition files:

| Relative to `DataRoot` | SHA-256 |
| --- | --- |
| `tactical/BAR-ATH_tactical_atd.xml` | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
| `tactical/BAR-ATH_tactical_atd.txt` | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |

These identities are copied from the existing frozen input provenance. The repository includes neither those files nor video, ground truth, raw tracking exports or training caches. The pretrained inference path does not require the private research repository.

The frozen Windows runtime is AMD64, CPython 3.11.0, with the exact nineteen distributions in `package/repro/requirements-windows-cp311.lock`. Core versions are NumPy 1.26.4, scikit-learn 1.3.2 and Torch 2.11.0+cu128. Its CUDA-enabled wheel is retained for runtime identity; inference uses CPU and four threads.

The outer setup tooling adds separate CPU profiles without changing the frozen model or package:

| Profile | Interpreter and platform | Dependency lock | Verification status |
| --- | --- | --- | --- |
| Windows | CPython 3.11.0, AMD64 | `package/repro/requirements-windows-cp311.lock` | Fresh Command Prompt setup and exact CSV reproduction passed |
| Linux | CPython 3.11.x, x86_64, glibc 2.28+ | `requirements/linux-cpu.lock` | Fresh setup and exact CSV reproduction passed on local Ubuntu WSL |
| macOS | CPython 3.11.x, Apple Silicon/arm64, macOS 12+ | `requirements/macos-arm64.lock` | All 19 wheels downloaded, hashes and metadata checked; execution untested |

Linux uses Torch 2.11.0+cpu; macOS uses its CPU-capable Torch 2.11.0 wheel. Both retain NumPy 1.26.4, pandas 1.5.2, scikit-learn 1.3.2 and SciPy 1.10.1. [requirements/SOURCES.json](requirements/SOURCES.json) records their wheel origins and hashes. Intel macOS and Linux ARM profiles are not included. A resolved wheel set does not establish successful model loading or exact CSV reproduction.

The Python workflows passed full reproduction from clean code commit `7f27085bfc9024391b22deda827b27ef1fc39f42` with newly installed pinned environments on Windows and local Ubuntu WSL. Both passed the official validator and produced the reference CSV bytes. Windows passed 20 release tests; Linux passed 16 with four PowerShell-specific tests skipped. The Linux environment was x86_64, glibc 2.43, CPython 3.11.18 and Torch 2.11.0+cpu. Both environments ran on the same physical host. [verification/PLATFORMS.md](verification/PLATFORMS.md) records their scope and receipts.

The macOS wheel download check verified all 154,559,425 bytes against recorded hashes, sizes and embedded metadata. It did not install or execute the wheels on a Mac; macOS execution remains untested.

`scripts/setup.py` needs no match data and records the selected profile, runtime and locked environment in the external run folder. `scripts/reproduce.py` creates a fresh setup or verifies an existing completed setup before inference. When setup uses an explicit `--python` interpreter, reproduction must receive the same path. It retains the frozen platform-native CSV and original receipt under `native-output/`, then writes the submission CSV and a separate receipt under `output/` using CRLF newlines. This serialization changes neither frame numbers nor labels. Every profile must match the same final reference hash; a mismatch is a failed run. The original PowerShell workflow and its evidence remain available.

**Original Windows verification, November 9, 2026.** A fresh Git checkout of commit `eba959d33ca8f3a1cd52ab4e997455e871e24c26`, matching the remote clone, reproduced the reference CSV in a newly installed environment with all nineteen pinned wheels. Dependency checks, official CSV format validation and all 13 release tests passed. Two independent CSV generations, with four inference passes in total, produced identical bytes; the deliverable CSV matched them.

The CSV has 199,319 ordered rows, is 1,882,093 bytes, and has SHA-256 `35d905533b696bee3f29da3e6c9f24ebab76780982e9f2325b6e67784dfea303`. That original verification covered Windows AMD64 with CPython 3.11.0; the separate later record adds Ubuntu WSL evidence on the same physical host. Neither establishes reproduction on another machine or every supported OS release. See [the reference identity](verification/reference.json), [original machine-readable evidence](verification/reproduction.json), and [original reproduction record](verification/REPRODUCTION.md).

The output receipt records the model, executable source, input and CSV hashes. CSV validation and reference-hash matching establish format and reproduction; they are not accuracy measurements.

## Rules and historical disclosure

The [November 3, 2026 organizer clarification](package/RULES_CLARIFICATION_2026-11-03.md) governs this candidate. All fitting and tuning for this rebuild used external data; its inference uses automatic tracking without video or target labels.

The preceding private research project included target-ground-truth tuning, provided-video fitting and a documented public-GT disclosure. The research snapshot disables the identified historical entry points and preserves their history. The new repository contains the externally rebuilt candidate; it does not establish that the historical project was always compliant or settle organizer eligibility. No publication or submission is implied by the existence of this local package.

The upstream competition source license and SkillCorner notice are retained in `package/LICENSE` and `package/THIRD_PARTY_NOTICES.md`. Those notices do not grant rights over the restricted competition match assets.
