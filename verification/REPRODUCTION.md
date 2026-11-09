# Reproduction verification

Reference output generated on 9 November 2026 for team **ochtagon**.

- CSV: `ochtagon.csv`, 1,882,093 bytes, 199,319 rows (frames 0 through 199318).
- SHA-256: `35d905533b696bee3f29da3e6c9f24ebab76780982e9f2325b6e67784dfea303`.
- Official format validator passed. Two inference passes produced identical labels.
- Model and source are the unchanged November 3 observation-gap package.
- Only authorized tactical tracking XML/TXT were used. No target ground truth,
  video, training, parameter selection or accuracy scoring was performed.

The reference identity is recorded in [reference.json](reference.json).
## Clean-checkout result: passed

Tested implementation commit: `eba959d33ca8f3a1cd52ab4e997455e871e24c26`.
A clean local clone and a separate clone downloaded from GitHub both resolved
to this commit. Their frozen package and reproduction scripts matched exactly.

The clean checkout's documented `scripts/reproduce.ps1` command created a new
CPython 3.11.0 Windows AMD64 environment, downloaded and installed all nineteen
hash-pinned distributions, passed `pip check`, and verified all 41 package payloads.
It then ran two full inference passes and the official CSV format validator.

The regenerated CSV matched the independently generated reference **byte for
byte**, including Windows line endings. Both files have the SHA-256 above.
The final upload copy was also compared byte for byte. Across the two independent
processes, four full inference passes completed with identical labels.

The full setup and reproduction command ran from 21:06:32 to 21:10:15 UTC on
9 November 2026 (about 3 minutes 42 seconds). This includes dependency installation;
it is not a standalone inference benchmark. All 13 release tests passed.
See [the machine-readable receipt](reproduction.json) for runtime versions,
input/model/source identities and explicit verification flags.

Verification establishes exact reproduction on the same Windows host in a fresh
checkout and newly installed environment. It does not establish other operating
systems, Python versions or hardware, retraining reproduction, prediction accuracy,
competition eligibility or qualification. Subsequent documentation/evidence commits
retain the exact tested package and script hashes recorded in the receipt.

## Installation issue found and fixed

The first rehearsal of commit `5d1350d` stopped during installation, before target
inference: the default C: temporary directory had only about 25 MB free. No source,
model, parameter or prediction change was made. The launcher was updated to place
temporary files and its explicit pip cache inside `RunRoot` and restore the calling
environment afterward. A subprocess regression verifies the temporary directory
and restoration on failure. A new checkout and entirely new run directory then
passed the complete rehearsal above. The failed local run is preserved separately;
it is not represented as a successful reproduction.

The package's internal evidence is immutable historical evidence. Its original
`target_inference_performed: false` fields and old installation-status comments
describe package creation and are superseded for later activity by this record.
