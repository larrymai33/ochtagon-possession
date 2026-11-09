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

The supported runtime is Windows AMD64, CPython 3.11.0, with the exact nineteen distributions in `package/repro/requirements-windows-cp311.lock`. Core versions are NumPy 1.26.4, scikit-learn 1.3.2 and Torch 2.11.0+cu128. Inference uses CPU and four threads. The CUDA-enabled wheel is retained for runtime identity, not a GPU requirement. Exact-file reproduction on other operating systems or Python versions is not established.

The reference CSV has 199,319 ordered rows and is 1,882,093 bytes. Its official format validation passed, and two inference passes produced identical labels. [verification/reference.json](verification/reference.json) records its hash and frame count. A fresh-checkout run in a newly installed environment is still pending; its outcome will be recorded in [verification/REPRODUCTION.md](verification/REPRODUCTION.md).

The output receipt records the model, executable source, input and CSV hashes. CSV validation and reference-hash matching establish format and reproduction; they are not accuracy measurements.

## Rules and historical disclosure

The [November 3, 2026 organizer clarification](package/RULES_CLARIFICATION_2026-11-03.md) governs this candidate. All fitting and tuning for this rebuild used external data; its inference uses automatic tracking without video or target labels.

The preceding private research project included target-ground-truth tuning, provided-video fitting and a documented public-GT disclosure. The research snapshot disables the identified historical entry points and preserves their history. The new repository contains the externally rebuilt candidate; it does not establish that the historical project was always compliant or settle organizer eligibility. No publication or submission is implied by the existence of this local package.

The upstream competition source license and SkillCorner notice are retained in `package/LICENSE` and `package/THIRD_PARTY_NOTICES.md`. Those notices do not grant rights over the restricted competition match assets.
