# Implementation review before external fitting

Recorded 2026-11-03. This is a source and synthetic-test review, not a model
performance result or organizer eligibility decision.

Independent reviewers checked the raw observation masking, paired-view trainer,
preparation, experiment contract, fold orchestration and final exporter. Review
findings about incomplete source bindings, preparation evidence rechecks and the
final fit-attempt receipt were repaired before external preparation began.

Substantive verification includes:

- Removing raw coordinates removes their influence through base, contact and
  temporal features. Inputs remain unchanged and masks stay inside valid runs.
- Zero augmentation, identical views and all-masked views reproduce the relevant
  original trainer exactly on synthetic CPU fixtures. Batch, optimizer, scheduler
  and team-swap schedules are retained; observation selection uses a separate RNG.
- A mocked complete fold sends exactly twelve training IDs to HGB, normalization
  and all four TCN fits. Held caches are loaded without labels. Every component
  predicts both held observation views, with independently checked blending.
- All sixteen frozen prediction files must verify before scoring opens held
  labels. Baseline metric reproduction precedes candidate scoring. Tampering and
  overwrite attempts fail.
- The final exporter requires a passing, recomputed similarity decision and
  bound predictions/source evidence before fitting. Synthetic serialization and
  native-frame inference round trips pass. Invalid identity/source evidence
  stops before target loading.

Focused results recorded before preparation: contract/scorer/fold tests 25 pass;
preparation tests 7 pass; final-export tests 15 pass. Earlier primitive/trainer
and related regression tests passed 75 checks. These groups overlap with the
broader suite and must not be summed as a unique test count.

The broader pinned-runtime run completed with **1,467 passed, 2 skipped and
1 deselected** in 276.30 seconds. The deselected case is the independently
reproduced failure below. The final exporter subsequently passed its current
15-case focused suite, including the last receipt-tampering regression. Ruff
passed for every new implementation/test module; whitespace checks passed.

The broader suite independently reproduced an existing Windows virtualenv issue
in `tests/test_benchmark_broadcast_runtime_v3.py::test_progress_tiny_worker_retains_completed_stages`:
the launcher PID differs from the worker journal PID. The worker completed and
its ten journal records validated using the actual worker PID. The benchmark and
test are byte-identical to starting commit `285aabe` and are outside the new model
pipeline. This is recorded as a test limitation, not a passing test. Their hashes:

- `scripts/benchmark_broadcast_runtime_v3.py`:
  `e95a3c405c0070c193da4957397382c03de0ccf7b2958978ff39d14f49b24cd0`
- `tests/test_benchmark_broadcast_runtime_v3.py`:
  `535b1965954341b510e57c51f091bdc6d8dd2b18c83de5274524dc997195a71a`

No target video, ground truth or manual labels were used in these reviews/tests.
External preparation and model-performance evidence will be recorded separately.
