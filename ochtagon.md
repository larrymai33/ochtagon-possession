# ochtagon: possession inference from automatic tracking

Team: **ochtagon**. Model: **observation_gap_v1**, frozen November 3, 2026.

Repository: [larrymai33/ochtagon-possession](https://github.com/larrymai33/ochtagon-possession).

## Approach

We infer home possession (`H`), away possession (`A`) and dead ball (`D`) from automatic tactical tracking. Frames outside the organizer-published match periods are `X`. The output contains one row for each of 199,319 tactical video frames.

Tracking and contact features describe player/ball geometry, motion, ball visibility and nearby interactions. A histogram gradient-boosting model and two temporal convolutional networks estimate live-play and home-team probabilities. Their fixed weights are 0.50, 0.25 and 0.25. Tracking is resampled within each valid run to 10 Hz, processed by the models, and projected back to the native 25 Hz timeline. A duration-constrained decoder turns those probabilities into possession intervals separately within each run.

We chose this approach because geometry supplies direct evidence of control while temporal context helps distinguish short observation failures from changes in possession. The candidate adds simulated gaps in external training ball observations to improve tolerance of missing tracking. It retains the established inference features, model architecture, ensemble weights and decoder.

## Training data and labels

The only labelled dataset used to train this packaged model is [SkillCorner Open Data](https://github.com/SkillCorner/opendata) at revision `4340d274572876239c154c90bc507a9b3250a656`. It supplies automatic tracking, match metadata and provider possession phases for other matches. The included attribution reproduces the upstream MIT notice.

The training cohort consists of these sixteen matches:

```text
1874553 1886347 1899585 1925299 1953632 1959846 1986691 1996435
2006229 2006363 2007448 2007721 2011166 2013725 2015213 2016236
```

Together they contain **945,508 valid frames at 10 Hz**. SkillCorner is the source of the automatic provider phase labels. The team did not independently hand-label this cohort, and these labels are not claimed to be independently adjudicated human ground truth. Code maps each provider phase's team and half-open frame interval to `H` or `A`; uncovered frames within a period become `D`, and frames outside periods become `X`. The provider's per-frame possession field is not the training label source.

External comparison used four folds grouped by whole match: twelve training matches and four held matches per fold. Normalization used original observations from the training fold only. Final fitting used all sixteen matches after the frozen comparison passed. Four previously inspected matches, `1927964`, `1996436`, `2010085` and `2017461`, remained quarantined and were not used in this rebuild.

The HGB model uses 120 iterations, learning rate 0.08, 63 leaves, minimum leaf size 30 and L2 regularization 1.0. Each TCN has 96 hidden channels, dilations 1, 2, 4, 8, 16, 32 and 64, dropout 0.1, and eight training epochs. Final seeds are 20261023 and 20261123. During temporal training, each window has a 0.5 probability of using a masked view. Masking removes raw ball observations in independently seeded intervals of 0.5, 1, 2 or 4 seconds, covering approximately 10% of valid frames, and recomputes the dependent features. Masks do not use possession labels or cross period boundaries.

## Possession definitions and assets

Loose balls and challenges retain the provider's team assignment while its phase continues. Restart timing is inferred from tracking evidence and the decoder. There is no manual target correction or frame-specific restart boundary. Decoder minimum durations are 0.1 seconds for `H` and `A`, and 2.5 seconds for `D`; team-switch and dead-switch penalties are 1.4 and 2.0. These settings were retained from documented external development.

Final inference uses only the supplied tactical EPTS tracking text and its automatic XML metadata, plus organizer-published match boundaries. It does not use target ground truth, tactical/broadcast video, Smart Tagging, homography or human event annotations. Earlier research explored video team-colour fitting, target-calibrated HSMM/Smart Tagging variants, broadcast ball fusion and alternative feature treatments. Those routes are excluded from this candidate. Metrica sample data and SPORTec were explored historically but did not train the packaged model.

## Results and limitations

The official ranking metric is spellF1, which averages possession-interval F1 over temporal IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7.

| External mean spellF1 | Retained ensemble | Candidate |
| --- | ---: | ---: |
| Ordinary observations | 0.751613 | 0.748792 |
| Fixed synthetic missing-ball gaps | 0.715707 | 0.729366 |

Ordinary mean and median paired changes were -0.002821 and -0.002023, both within the predeclared -0.005 similarity limits. Ordinary results improved on four of sixteen matches; masked results improved on fifteen. This is a tradeoff, with slightly weaker ordinary performance and better behavior under the controlled gaps.

These sixteen matches were repeatedly used for development. Their results are not an independent generalization estimate. Synthetic gaps do not reproduce every real camera or tracking failure. Provider phase definitions may differ from the competition's controlled-possession definition, particularly around loose balls, challenges and restarts. Short possessions remain difficult: at diagnostic IoU 0.5, the ordinary candidate recovered 652 of 1,519 external possessions shorter than five seconds. We do not infer a target full-match score or qualification probability from these measurements. Further improvement needs fresh permitted external matches with compatible, independently checked labels.

## Reproduction and history

Follow [README.md](README.md) for the platform requirements and shell examples. Windows AMD64 uses CPython 3.11.0; Linux x86_64 and Apple Silicon macOS CPU profiles use CPython 3.11.x. The Python workflows passed exact CSV reproduction and official format validation from clean commit `7f27085bfc9024391b22deda827b27ef1fc39f42` with fresh environments on Windows and local Ubuntu WSL, on the same physical host. The WSL run used glibc 2.43, CPython 3.11.18 and Torch 2.11.0+cpu. Windows passed 20 release tests; Linux passed 16 with four PowerShell-specific tests skipped. All nineteen macOS wheels passed download, hash and metadata checks, but macOS hardware execution remains untested. See [current platform evidence](verification/PLATFORMS.md).

Setup can be completed without match data, using `scripts/setup.py --run-root <new-folder>`. To create the environment and generate the CSV in one command on Windows, run:

```text
python scripts/reproduce.py --run-root "E:/ochtagon-run" --data-root "E:/authorized-data" --profile windows
```

On Linux or macOS, use `python3.11`, absolute local paths and the automatic platform selection. Both scripts accept `--python <interpreter>` and `--profile auto|windows|linux|macos`; repeat an explicit setup `--python` path when running reproduction. Reproduction can reuse a verified completed setup without activating its environment. The original `scripts/reproduce.ps1` Windows route remains available and requires its own new, absent run folder.

The workflow installs the hash-pinned environment, verifies the frozen package, runs inference twice and validates `output/ochtagon.csv` against [the recorded reference](verification/reference.json). The Python wrapper preserves the original CSV and inference receipt under `native-output/`; final serialization uses CRLF newlines without altering frames or labels. Every platform must match the same reference hash. [The reproduction record](verification/REPRODUCTION.md) identifies the original tested checkout and exact-file checks. The workflow does not retrain, fit on the provided game, score target ground truth or compare output with video. The model SHA-256 is `76c4508ff2091e008de23ddec32ecfc86f989e172916702cadd37511dc68b672`.

The original Windows PowerShell verification on November 9, 2026 used a fresh checkout of commit `eba959d33ca8f3a1cd52ab4e997455e871e24c26` and a newly installed pinned environment. Exact CSV reproduction, official format validation and all 13 release tests passed. The same 199,319-row, 1,882,093-byte CSV was reproduced by the later Windows and WSL runs; its SHA-256 is `35d905533b696bee3f29da3e6c9f24ebab76780982e9f2325b6e67784dfea303`. These checks establish exact reproduction in the tested environments, not target accuracy or portability to another machine.

This candidate's fitting, normalization and parameter-selection lineage use the declared external data. The [provenance review](package/evidence/PROVENANCE_REVIEW.md) documents the evidence and its limits. Earlier work in the private research project included target-ground-truth tuning, provided-video fitting and a documented public-GT disclosure. The November 3 organizer clarification superseded older permissions, and historical entry points were disabled. Creating this repository and externally rebuilding the model do not erase those activities or establish organizer eligibility. The private research snapshot is commit `8017a0a46ff873b71084a5396ece2980ac14a708`; its data and history are not required to run the packaged inference.
