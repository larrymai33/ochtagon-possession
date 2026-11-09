# Independent aggregate result review

Reviewed 2026-11-03. Repository root: `E:\GitHub\more-than-a-hack-2027`.

**The candidate passes the predeclared engineering similarity criterion. Its
ordinary-view spellF1 is slightly worse, while performance under the fixed
synthetic observation-gap stress test is better. This is not evidence that the
candidate is stronger overall.**

Only `result.json`, `freeze.json`, `contract.json`, `preparation.json`, and
`preparation-freeze.json` in this directory were inspected for this result
review. Arithmetic below was independently recomputed from their aggregate
per-match scores and diagnostic counts using Python's standard library. No
labels, prediction arrays, raw matches, target assets, or model artifacts were
opened; no scoring against labels, training, or tuning was performed.

## Verified identities and consistency

Actual file SHA-256 values:

- `result.json`: `d750c868d400695a88438f36081f9ff4a1a23c4538762e85534325a0e42c43c4`
- `freeze.json`: `8d9bbadf8f1bc96e24374a0893ceaf38d8d73239575a0758a3c6cfdb8d1e0af9`
- `contract.json`: `8c00114c32c09d9d8fc24aa145458fafb13883c17735b35b88dc7a28d98185ab`
- `preparation.json`: `efceb11824f79ad163ac7b7373ad6a6c3048c9a77147a25c894f95c42b79e741`
- `preparation-freeze.json`: `78c7c527d29e0420d555fcd3abea20e48f54844160c28bcab03a4f1c03381c40`

The first three hashes match the identities supplied for review. The result
references the verified prediction-freeze hash. Result and freeze both bind
canonical contract payload SHA-256
`6169214d04a5884d9273183b36d46482eb33eba8fb5e41d85c6a16ac4efa6615`.
Preparation hashes, mask configuration, all sixteen IDs, 139 preparation-source
bindings, sixteen original-cache hashes, and sixteen prepared-view hashes are
consistent across the manifests. All saved per-match delta fields equal the
differences of their component scores.

The freeze covers exactly the four declared folds and sixteen held matches.
All 48 recorded baseline component maximum probability errors are **0.0**, and
all sixteen recorded decoded-prediction comparisons are equal. Each fold records
both seeds for baseline and candidate, eight epochs each, identical learning-rate
sequences, and identical batch counts: 172/173/172/172 batches per epoch in folds
0/1/2/3. Each fold records its normalizer hash. These are audited receipts;
prediction and model bytes were deliberately not reopened in this review.

## Independently recomputed comparison

| Measure | Ordinary observations | Fixed masked observations |
|---|---:|---:|
| Baseline mean spellF1 | 0.7516131654094745 | 0.7157069872466137 |
| Candidate mean spellF1 | 0.7487920882860035 | 0.7293662293222551 |
| Mean paired change | -0.002821077123470926 | +0.013659242075641308 |
| Median paired change | -0.0020225497863962105 | +0.016512961938667248 |
| Wins / ties / losses | 4 / 0 / 12 | 15 / 0 / 1 |
| Baseline mean macroF1 | 0.9193689027048576 | 0.9040356585598820 |
| Candidate mean macroF1 | 0.9193240136575981 | 0.9119378906065827 |

The frozen gate requires baseline verification plus ordinary-view mean and median
paired changes each at least -0.005 (numeric tolerance 1e-12). Both pass without
needing that tolerance. Independently recomputing the predicate returns `true`,
agreeing with the saved result. This gate does not require a majority of wins and
does not establish improvement or statistical noninferiority. Five ordinary-view
matches decline by more than 0.005; the largest decline is -0.013043478260869379.

The approximately 9e-17 difference between the independently averaged masked
paired change and the saved difference of means is floating-point summation
roundoff, not a substantive discrepancy.

| Match | Ordinary spellF1 change | Masked spellF1 change |
|---|---:|---:|
| 1874553 | -0.006844 | +0.020260 |
| 1886347 | +0.005276 | +0.026835 |
| 1899585 | -0.001210 | +0.017721 |
| 1925299 | -0.002320 | +0.012041 |
| 1953632 | -0.002716 | +0.021493 |
| 1959846 | +0.002646 | +0.020877 |
| 1986691 | +0.001029 | +0.019775 |
| 1996435 | -0.005963 | +0.009125 |
| 2006229 | -0.005445 | +0.026560 |
| 2006363 | -0.013043 | -0.006490 |
| 2007448 | +0.000741 | +0.015305 |
| 2007721 | -0.001726 | +0.001199 |
| 2011166 | -0.001606 | +0.007185 |
| 2013725 | -0.002883 | +0.002716 |
| 2015213 | -0.000869 | +0.018888 |
| 2016236 | -0.010205 | +0.005056 |

## Diagnostic tradeoffs

Diagnostics use same-team, one-to-one matching at IoU 0.5. Short means strictly
less than five seconds; false-insertion rate is unmatched predictions divided
by all predicted possessions. Counts and reported rates reconcile.

| Diagnostic | Ordinary baseline -> candidate | Masked baseline -> candidate |
|---|---|---|
| Short spells recovered, out of 1,519 | 664 -> 652 | 584 -> 586 |
| Short-spell recall | 43.713% -> 42.923% | 38.446% -> 38.578% |
| Matched possessions, out of 4,125 | 2,953 -> 2,939 | 2,796 -> 2,827 |
| False insertions | 668 -> 682 | 758 -> 688 |
| Predicted possessions | 3,621 -> 3,621 | 3,554 -> 3,515 |
| False-insertion rate | 18.448% -> 18.835% | 21.328% -> 19.573% |

Ordinary observations therefore show modestly worse short-spell recall and
false-insertion behavior. Under masking, there are 70 fewer false insertions
and 31 more matched possessions, but only two additional short spells recovered.
The stress result should not be described as a substantial short-possession
recovery improvement. Masked merge-prediction flags also rise from 234 to 249,
while split-ground-truth flags fall from 153 to 119; these descriptive flags
overlap and are not causal explanations. Mean absolute boundary errors improve
under masking (start: 1.1169 -> 1.1052 seconds; end: 1.1602 -> 1.0785 seconds),
and are slightly worse under ordinary observations.

Across 945,508 valid frames, the fixed stress mask schedules 94,955 frames
(10.04275%). It removes 56,727 detected finite ball observations (5.99963% of
valid frames) and 70,670 finite ball coordinates (7.47429%). Existing missingness
explains the difference between scheduled coverage and effective removal.
The result's coverage fields agree with preparation counts and denominators.

## Interpretation and limits

Supported conclusion: **the new training approach preserves mean and median
external-development performance within the fixed similarity margin and is less
sensitive to this particular synthetic ball-observation-gap condition.** It
trades a small ordinary-view decline for better fixed-stress performance.

The stress mask is one fixed deterministic protocol, and labels come from the
same external provider. The sixteen matches were repeatedly consumed during
development. These results do not demonstrate stronger overall performance,
independent holdout generalization, target-game accuracy, or organizer
eligibility. This review also makes no claim about completion or verification
of the separate final sixteen-match fit.
