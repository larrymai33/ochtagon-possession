# External observation-gap ensemble

This local pretrained candidate meets the frozen external similarity criterion.
It is slightly weaker on ordinary inputs and improves the controlled missing-ball
test. It is not an overall accuracy improvement or an organizer eligibility decision.

| External mean spellF1 | Retained ensemble | New candidate |
| --- | ---: | ---: |
| Ordinary observations | 0.751613 | 0.748792 |
| Scheduled observation gaps | 0.715707 | 0.729366 |

Ordinary mean change: -0.002821; median paired change:
-0.002023. Both pass the predeclared -0.005 limits.
There were 4 ordinary-view wins out of 16 matches.
The sixteen matches were repeatedly used for development, not an independent test.
No target score or output/video comparison was used for this candidate.

The architecture and inference features remain HGB plus two temporal networks,
weighted 0.50/0.25/0.25. Temporal fitting adds independently seeded raw ball gaps
on external training matches, recomputes dependent features, and chooses original
or masked windows with probability 0.5. Final fitting uses all sixteen permitted
external matches. Training uses original-only normalization and the retained
epoch, seed, optimizer and decoder settings.

## Pretrained inference

Use the pinned Windows Python 3.11 runtime described by the files under `repro`.
No match data is included. The source directory contains the complete inference
closure; retraining requires the research repository and its bound external data.
Run from this package's `source` directory, with an explicit permitted tracking
data directory and a new output CSV path:

```powershell
python -m solution.final_observation_ensemble infer --model ../models/observation_gap_v1.pkl --expected-sha256 76c4508ff2091e008de23ddec32ecfc86f989e172916702cadd37511dc68b672 --data-dir E:/path/to/automatic-tracking --out E:/path/to/new-output.csv
```

The command validates the artifact and source before loading automatic tracking,
runs inference twice to check reproducibility, and creates a CSV and hash receipt
without overwriting existing outputs. It does not fit a model or load GT/video.
Do not use target scores or output/video comparison to tune the model.

Model SHA-256: `76c4508ff2091e008de23ddec32ecfc86f989e172916702cadd37511dc68b672`.
Evidence is under `evidence`; `repro/package-manifest.json` binds package files.
Historical violations remain disclosed in the research repository. This repair
does not erase them or establish competition eligibility. The prior accepted
package remains unchanged. Nothing has been published or submitted.
