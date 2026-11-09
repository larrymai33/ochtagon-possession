# Retained-setting provenance and claims review

Reviewed 2026-11-03. Repository root: `E:\GitHub\more-than-a-hack-2027`.

This review used executable source, methodology/provenance documentation, and
bounded aggregate metadata from the four external reports listed below. It did
not access target GT, video, manual labels, target predictions, raw external
matches, model artifacts, or the running candidate's scores. No additional
training, scoring, or tuning was performed.

## Conclusion

No retained HGB, TCN, blend, or duration-decoder setting was found to have been
demonstrably selected on target GT or video. The reviewed records document an
external-development selection chain, and the current candidate's fitting,
normalization, and comparison paths use only the sixteen allowed external
matches. This supports a claim about the current experiment and its documented
selection lineage. It does not establish that all historical project work was
external-only or resolve eligibility after the acknowledged earlier violations.

## Verified external selection evidence

The following paths are relative to the repository root. Their actual file
SHA-256 values were checked against the identities named in retained source and
documentation; only aggregate/configuration/provenance fields were inspected.

1. `data/cache/external_model/experiments/hgb_contact_context_segment_balanced_capacity_v5.json`

   SHA-256: `7d7f3738387c4962583275b7b41bad68bc19144867e7944ec074baab0bf19411`

   Records 16 external development matches, 253 features, stride 5, and HGB
   settings learning rate 0.08, 120 iterations, 63 leaves, minimum leaf size 30,
   L2 1.0, seed 20261023. Its two recorded decoder comparisons select minimum
   H/A/D durations 0.1/0.1/2.5 seconds and team/dead penalties 1.4/2.0:
   external mean spellF1 0.7294285490338521 versus 0.7237888646638195 for the
   recorded 0.1/0.1/2.0-second, 1.0/2.0 alternative. The match IDs exactly match
   the current allowed cohort. `METHODOLOGY.md` section 10.4 documents the
   external feature, weighting, and capacity comparisons.

2. `data/cache/external_model/experiments/high_capacity_contact_tcn_gate_v1_a906698e7d1c_7e27f287abec.json`

   SHA-256: `4f2c044aa4ab87a3739c22c629f98646c9ddfdbaccc9ad9d189e749e5447131c`

   Records the 96-channel, seven-dilation, eight-epoch TCN configuration and
   an explicit external-only gate contract: twelve training caches plus match
   1874553, excluding target labels/public GT and quarantined matches. The
   source definition is `solution/evaluate_high_capacity_tcn_gate.py`.

3. `data/cache/external_model/experiments/high_capacity_contact_tcn_full_oof_v1_543780d80314_c51bf984e03d.json`

   SHA-256: `8c303b496e4bf6f705ecaaa4fc5ea0aef09df74bf6b49ce99dbf7435b2d194b7`

   Records the accepted four-fold TCN confirmation, fixed 50/50 arithmetic HGB
   blend, canonical HGB decoder, fold seeds 20261023 through 20261026, and
   `no_grid_or_retuning: true`. The accepted mean external spellF1 is
   0.7492213923134956. Source/input hashes and the external data-access contract
   are recorded. See `METHODOLOGY.md` section 10.21 and
   `solution/evaluate_high_capacity_tcn_full_oof.py`.

4. `data/cache/external_model/experiments/high_capacity_contact_tcn_seed_ensemble_v1_a1556e75758d_d42b1498e21c.json`

   SHA-256: `486d5a6f274c1e17b2cab4fe9302ec2288ffff90debdd7a380b141de5bed6cb2`

   Records the accepted second-seed experiment, a fixed +100 seed offset,
   arithmetic weights 0.50/0.25/0.25, and the unchanged canonical decoder. The
   accepted mean external spellF1 is 0.7516131654094745. The report binds its
   predecessor, component probabilities, and all sixteen training caches.
   See `METHODOLOGY.md` section 10.24 and
   `solution/evaluate_high_capacity_tcn_seed_ensemble.py`.

## Current candidate and historical separation

`solution/observation_experiment_contract.py` freezes the retained settings,
whole-match folds, original-only training-fold normalization, independently
seeded raw ball gaps, 0.5 paired-view probability, and similarity criterion.
`solution/evaluate_observation_robustness.py` fits the HGB and both temporal
models using only each fold's twelve external training matches.
`solution/final_observation_ensemble.py` gates final fitting on the frozen
external comparison and refits on the sixteen external matches. It does not
warm-start from a target-trained model.

The retained numerical decoder is `solution/temporal.py`; the audited candidate
path does not import the historical `solution/decode.py` parameter/trust tables
or `solution/team_colour.py` video fitting. Those older workflows are explicitly
disabled and remain historical evidence. `docs/target-bootstrap-075-plan.md`
and `METHODOLOGY.md` section 10.49 describe the retained ensemble's historical
target check as subsequent to its external selection/freeze, not as a source
of new parameter values. That is documentary evidence, not erasure of earlier
target-GT/video violations elsewhere in the repository.

## Limits and appropriate wording

The early HGB-v5 report has no dedicated source/input provenance block or
pre-score contract. Its recorded external comparisons, methodology entry, and
later hash-bound references support the stated lineage, but cannot prove the
absence of every unrecorded historical influence. No specific inherited
target-selected setting was identified; treating this provenance gap as proof
of contamination would be speculation.

The sixteen matches have been repeatedly used for development. Neither their
historical scores nor the current similarity comparison is an independent
holdout estimate, a guarantee of target performance, or an organizer eligibility
decision. Older documentation allowing target-data fitting is superseded by
`RULES_CLARIFICATION_2026-11-03.md`.

Supported wording: **“This experiment uses only the declared external data for
model fitting, normalization, and parameter selection, and retains settings
documented as selected on external development matches. Historical violations
remain disclosed; the repair does not establish organizer eligibility.”**

Avoid claiming that the entire project was always compliant, that prior target
exposure has been undone, or that an independent target-performance result has
been established.
