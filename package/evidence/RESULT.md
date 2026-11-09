# External observation-gap comparison

Recorded 2026-11-03. The candidate passes the predeclared engineering similarity
criterion. Ordinary performance is slightly lower; performance with the fixed
observation gaps is higher. This is a tradeoff, not an overall accuracy improvement.

| Mean spellF1 | Retained ensemble | Candidate | Change |
| --- | ---: | ---: | ---: |
| Ordinary inputs | 0.751613165 | 0.748792088 | -0.002821077 |
| Masked inputs | 0.715706987 | 0.729366229 | +0.013659242 |

Ordinary median paired change: -0.002022550; 4 wins / 12 losses.
Both ordinary mean and median changes exceed the fixed -0.005 limits.
Masked inputs improve on 15 of 16 matches. This is the scheduled synthetic
missing-observation condition on external data, not a new game-level holdout.

All sixteen retained decoded predictions reproduced exactly. All 48 recorded
component probability maximum errors are zero. All four views of all sixteen
held matches were sealed before scoring; baseline scores reproduced before
candidate scores were calculated. Source/input/runtime checks passed.

## Diagnostic tradeoffs

| View | Short spells recovered / total (<5 s) | Recovery | Unmatched predictions / all predictions | Rate |
| --- | ---: | ---: | ---: | ---: |
| baseline | 664 / 1519 | 43.713% | 668 / 3621 | 18.448% |
| candidate | 652 / 1519 | 42.923% | 682 / 3621 | 18.835% |
| baseline_masked | 584 / 1519 | 38.446% | 758 / 3554 | 21.328% |
| candidate_masked | 586 / 1519 | 38.578% | 688 / 3515 | 19.573% |

Matching uses same-team IoU 0.5 and 10 FPS. The masked improvement mainly
reduces false insertions; short-spell recovery increases by only two spells.
Ordinary short-spell recovery decreases by twelve spells.

## Per-match spellF1

| Match | Retained | Candidate | Change | Masked retained | Masked candidate | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1874553 | 0.787879 | 0.781034 | -0.006844 | 0.733043 | 0.753304 | +0.020260 |
| 1886347 | 0.771300 | 0.776577 | +0.005276 | 0.730631 | 0.757466 | +0.026835 |
| 1899585 | 0.741520 | 0.740310 | -0.001210 | 0.709980 | 0.727701 | +0.017721 |
| 1925299 | 0.720332 | 0.718012 | -0.002320 | 0.692600 | 0.704641 | +0.012041 |
| 1953632 | 0.769383 | 0.766667 | -0.002716 | 0.703759 | 0.725253 | +0.021493 |
| 1959846 | 0.716569 | 0.719216 | +0.002646 | 0.672510 | 0.693387 | +0.020877 |
| 1986691 | 0.732892 | 0.733921 | +0.001029 | 0.688539 | 0.708315 | +0.019775 |
| 1996435 | 0.777391 | 0.771429 | -0.005963 | 0.737778 | 0.746903 | +0.009125 |
| 2006229 | 0.760802 | 0.755357 | -0.005445 | 0.709620 | 0.736180 | +0.026560 |
| 2006363 | 0.795652 | 0.782609 | -0.013043 | 0.757709 | 0.751220 | -0.006490 |
| 2007448 | 0.711848 | 0.712589 | +0.000741 | 0.681132 | 0.696437 | +0.015305 |
| 2007721 | 0.771726 | 0.770000 | -0.001726 | 0.752941 | 0.754140 | +0.001199 |
| 2011166 | 0.752610 | 0.751004 | -0.001606 | 0.713469 | 0.720654 | +0.007185 |
| 2013725 | 0.752542 | 0.749660 | -0.002883 | 0.746939 | 0.749655 | +0.002716 |
| 2015213 | 0.716697 | 0.715827 | -0.000869 | 0.698378 | 0.717266 | +0.018888 |
| 2016236 | 0.746667 | 0.736462 | -0.010205 | 0.722282 | 0.727338 | +0.005056 |

## Scope and reproducibility

The comparison uses sixteen repeatedly consumed external development matches,
four whole-match folds and the retained HGB/two-TCN architecture. The new method
recomputes features after raw, label-independent ball-observation gaps, then
chooses original or masked windows with probability 0.5. Normalization uses only
original training-fold observations. The optimizer schedule, seeds and decoder
are unchanged. No target GT, video, manual labels or target score was used.

The prepared cohort contains 945,508 valid frames; 94,955 frames were scheduled
for gaps, 56,727 detected ball observations and 70,670 finite ball observations
were removed. Original features reproduced the accepted cache exactly for all
sixteen matches. No quarantined external matches were accessed.

The final candidate is fitted on all sixteen external matches after this gate.
See [final fit](../observation_robustness_final_v1/fit.json) and the
[local inference package](../../release/observation_gap_v1/README.md).

These results do not establish target-game performance, independent generalization,
competition qualification or restored eligibility. Historical violations remain
disclosed. The retained model and package are unchanged. Nothing was published.

- [Independent result review](RESULT_REVIEW.md).
- [Setting provenance and historical limits](PROVENANCE_REVIEW.md).
- [Implementation review and test limitations](IMPLEMENTATION_REVIEW.md).
- Contract SHA-256: `8c00114c32c09d9d8fc24aa145458fafb13883c17735b35b88dc7a28d98185ab`.
- Freeze SHA-256: `8d9bbadf8f1bc96e24374a0893ceaf38d8d73239575a0758a3c6cfdb8d1e0af9`.
- Result SHA-256: `d750c868d400695a88438f36081f9ff4a1a23c4538762e85534325a0e42c43c4`.
